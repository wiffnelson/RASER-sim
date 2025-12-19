import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# --- System Parameters ---
# These parameters are now defined in the main scope so they can be
# accessed by both the 'system' function and the initial conditions.
Gamma = 0.4   # Pumping rate
d1_0 = 2e17   # Starting value for d1
d2_0 = 2e17   # Starting  value for d2

# Define the system of 6 coupled ODEs with named variables
def system(t, y):
    d1, d2, a1, a2, phi1, phi2 = y

    T1 = 15.0
    T2 = 0.13
    coupling_beta = 6.44662e-16
    deltaNu = 7.1 # Distance (in Hz) between peaks
    nu0 = 1e6
    epsilon = 1e-20
    f_off = 15 # frequency offset to move both signals positive

    # The dd1_dt and dd2_dt equations have been updated to include a pumping term.
    dd1_dt = Gamma * (d1_0 - d1) - (d1 / T1) - 4 * coupling_beta * (a1**2 + a1 * a2 * np.cos(phi1 - phi2))
    dd2_dt = Gamma * (d2_0 - d2) - (d2 / T1) - 4 * coupling_beta * (a2**2 + a2 * a1 * np.cos(phi1 - phi2))

    da1_dt = -(a1 / T2) + coupling_beta * d1 * (a1 + a2 * np.cos(phi1 - phi2))
    da2_dt = -(a2 / T2) + coupling_beta * d2 * (a2 + a1 * np.cos(phi1 - phi2))

    dphi1_dt = 2 * np.pi * (nu0 + f_off + (deltaNu / 2)) + coupling_beta * (d1 / max(a1,epsilon)) * a2 * np.sin(phi2 - phi1)
    dphi2_dt = 2 * np.pi * (nu0 + f_off - (deltaNu / 2)) + coupling_beta * (d2 / max(a2,epsilon)) * a1 * np.sin(phi2 - phi1)

    return [dd1_dt, dd2_dt, da1_dt, da2_dt, dphi1_dt, dphi2_dt]

# Time span and initial conditions
t_f = 16
t_span = (0, t_f)
t_eval = np.linspace(*t_span, 200 * t_f)
initial_conditions = [d1_0, d2_0, 1e10, 1e10, np.pi/2, np.pi/3]  # [d1, d2, a1, a2, phi1, phi2]


print("Starting ODE Solver...")
start = time.perf_counter()
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='BDF')
end = time.perf_counter()
print("ODE Solver done")
print(f"{end - start:.4f} seconds elapsed")
print(solution.message)

# Extract time and variables
t = solution.t
d1, d2, a1, a2, phi1, phi2 = solution.y

# Construct output signal: coherent field from a1*exp(i*phi1) + a2*exp(i*phi2)
output_signal = (1 / np.sqrt(2)) * (a1 * np.real(np.exp(1j * phi1)) + a2 * np.real(np.exp(1j * phi2)))

# Global variables to store the plots, allowing us to update them.
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
zoomed_plot_line = None

# --- Interactive Functionality ---

def on_select(eclick, erelease):
    """
    Callback function to handle the selection event.
    eclick is the press event, erelease is the release event.
    """
    # Clear previous zoomed-in plot to avoid clutter
    global zoomed_plot_line
    if zoomed_plot_line:
        zoomed_plot_line.pop(0).remove()
        fig.canvas.draw_idle()

    # Get the x-coordinates of the selected rectangle
    x1, x2 = eclick.xdata, erelease.xdata
    
    # Ensure x1 is smaller than x2
    if x1 > x2:
        x1, x2 = x2, x1

    # Find the indices corresponding to the selected time range
    indices = np.where((t >= x1) & (t <= x2))[0]
    
    if len(indices) < 2:
        print("Selection too small. Please select a wider range.")
        return

    # Get the corresponding data for the selected range
    t_zoomed = t[indices]
    output_signal_zoomed = output_signal[indices]

    # Compute Fourier Transform of the zoomed-in signal
    freq_zoomed = np.fft.fftfreq(len(t_zoomed), d=(t_zoomed[1] - t_zoomed[0]))
    Y_zoomed = np.fft.fft(output_signal_zoomed)
    positive_freqs_zoomed = freq_zoomed > 0

    # Plot the new FFT on the frequency domain graph, in a different color
    zoomed_plot_line = ax[1].plot(freq_zoomed[positive_freqs_zoomed], np.abs(Y_zoomed[positive_freqs_zoomed]), 
                                  color='red', linestyle='--', label='Zoomed-in FFT')

    # Update the legend to show the new line
    ax[1].legend()
    fig.canvas.draw_idle()

# --- Plotting Setup ---

# Time domain plot (original)
max_abs_y = np.max(np.abs(output_signal))
padding = max_abs_y * 0.05
ax[0].plot(t, output_signal, label='Original Output Signal')
ax[0].set_title('Time Domain: Select Region to Analyze')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)
ax[0].legend()

# Frequency domain plot (original)
freq = np.fft.fftfreq(len(t) , d=(t[1] - t[0]))
Y = np.fft.fft(output_signal)
positive_freqs = freq > 0
ax[1].plot(freq[positive_freqs], np.abs(Y[positive_freqs]), label='Original Full FFT')
ax[1].set_title('Frequency Domain: Full and Zoomed FFT')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Magnitude')
ax[1].grid(True)
ax[1].legend()
# ax[1].set_xlim(0, 100) # 👈 Add this line to manually control the x-axis limits

# Create the RectangleSelector widget and connect it to the time domain plot
rect_selector = RectangleSelector(ax[0], on_select,
                                  useblit=True,
                                  interactive=True)

plt.tight_layout()
plt.show()

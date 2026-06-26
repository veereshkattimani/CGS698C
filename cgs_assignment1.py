# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import poisson

# # Parameter
# lambda_val = 10

# # Range of x values (0 to 50)
# x = np.arange(0, 51)

# # Compute Poisson PMF
# pmf = poisson.pmf(x, lambda_val)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.bar(x, pmf, width=0.8, color='skyblue', edgecolor='black', alpha=0.7)
# plt.xlabel('Number of road accidents per day (x)', fontsize=12)
# plt.ylabel('Probability P(X = x)', fontsize=12)
# plt.title(f'Poisson Probability Mass Function (λ = {lambda_val})', fontsize=14)
# plt.xticks(np.arange(0, 51, 5))
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Given data
x_single = 220
x_sample = np.array([303, 443, 220, 560, 880])

# Define the log-normal PDF (sigma = 1) as a function of mu, with fixed x
def likelihood_single(mu, x):
    """Likelihood L(mu | x) for a single observation x."""
    return (1 / (x * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / 2)

# Define the joint likelihood for the full sample
def likelihood_joint(mu, x_array):
    """Joint likelihood L(mu | x1..xn) for the sample."""
    n = len(x_array)
    log_lik = -n * np.log(np.sqrt(2 * np.pi)) - np.sum(np.log(x_array)) - 0.5 * np.sum((np.log(x_array) - mu)**2)
    return np.exp(log_lik)  # return actual likelihood (very small numbers)

# ------------------------------
# (a) Likelihood for x = 220
# ------------------------------
mu_range = np.linspace(4, 7, 500)  # reasonable range around ln(220)≈5.39
L_single = [likelihood_single(mu, x_single) for mu in mu_range]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu_range, L_single, 'b-', linewidth=2)
plt.xlabel('μ')
plt.ylabel('Likelihood L(μ | x=220)')
plt.title('(a) Likelihood function for a single observation x = 220')
plt.grid(True, alpha=0.3)

# ------------------------------
# (b) Likelihood for the full sample
# ------------------------------
L_joint = [likelihood_joint(mu, x_sample) for mu in mu_range]

plt.subplot(1, 2, 2)
plt.plot(mu_range, L_joint, 'r-', linewidth=2)
plt.xlabel('μ')
plt.ylabel('Likelihood L(μ | all data)')
plt.title('(b) Likelihood function for the full sample')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# (c) Find μ that maximises likelihood (MLE)
# ------------------------------
# Method 1: analytical MLE = mean of log(x_i)
log_means = np.mean(np.log(x_sample))
print(f"\nAnalytical MLE (mean of logs): μ = {log_means:.6f}")

# Method 2: numerical optimisation (for verification)
def neg_log_likelihood(mu, x_array):
    """Negative log-likelihood for minimisation."""
    n = len(x_array)
    return n * np.log(np.sqrt(2 * np.pi)) + np.sum(np.log(x_array)) + 0.5 * np.sum((np.log(x_array) - mu)**2)

result = minimize_scalar(neg_log_likelihood, args=(x_sample,), bounds=(5, 7), method='bounded')
print(f"Numerical MLE: μ = {result.x:.6f}")

# Optional: plot the maximum point on the likelihood graph
plt.figure(figsize=(8, 5))
plt.plot(mu_range, L_joint, 'r-', linewidth=2, label='Likelihood')
plt.axvline(x=log_means, color='black', linestyle='--', label=f'MLE μ = {log_means:.4f}')
plt.xlabel('μ')
plt.ylabel('Likelihood')
plt.title('(c) Maximum likelihood estimate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
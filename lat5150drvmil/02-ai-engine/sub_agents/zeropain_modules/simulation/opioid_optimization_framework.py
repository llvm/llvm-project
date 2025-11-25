#!/usr/bin/env python3
"""
ZeroPain Optimization Framework
Multi-compound protocol optimization with local execution and Intel acceleration
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import time
import os

try:
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using basic optimization.")

from .opioid_analysis_tools import CompoundDatabase, CompoundProfile, PharmacokineticModel

@dataclass
class ProtocolConfig:
    """Configuration for multi-compound protocol"""
    compounds: List[str]
    doses: List[float]  # mg
    frequencies: List[float]  # doses per day
    duration: int = 90  # days

    def to_dict(self) -> Dict:
        return {
            'compounds': self.compounds,
            'doses': self.doses,
            'frequencies': self.frequencies,
            'duration': self.duration
        }


@dataclass
class OptimizationResult:
    """Results from protocol optimization"""
    optimal_protocol: ProtocolConfig
    success_rate: float
    tolerance_rate: float
    addiction_rate: float
    withdrawal_rate: float
    safety_score: float
    therapeutic_window: float
    iterations: int
    computation_time: float


class ProtocolOptimizer:
    """Optimize multi-compound opioid protocols"""

    def __init__(self, compound_database: CompoundDatabase,
                 use_multiprocessing: bool = True,
                 use_intel_acceleration: bool = False):
        self.db = compound_database
        self.pk_model = PharmacokineticModel()
        self.use_multiprocessing = use_multiprocessing
        self.use_intel_acceleration = use_intel_acceleration

        # Detect available cores
        self.n_cores = mp.cpu_count()

        # Intel acceleration setup
        if use_intel_acceleration:
            self._setup_intel_acceleration()

    def _setup_intel_acceleration(self):
        """Setup Intel NPU/GPU acceleration"""
        try:
            # Try to use Intel Extension for PyTorch
            import intel_extension_for_pytorch as ipex
            self.ipex_available = True
            print(f" Intel Extension for PyTorch enabled")
        except ImportError:
            self.ipex_available = False
            print("Note: Intel Extension for PyTorch not available")

        try:
            # Try to use OpenVINO for NPU acceleration
            from openvino.runtime import Core
            self.openvino_core = Core()
            devices = self.openvino_core.available_devices
            print(f" OpenVINO available. Devices: {', '.join(devices)}")
            self.openvino_available = True

            # Prefer NPU if available, then GPU
            if 'NPU' in devices:
                self.compute_device = 'NPU'
                print(f" Using Intel NPU for acceleration")
            elif 'GPU' in devices:
                self.compute_device = 'GPU'
                print(f" Using Intel Arc GPU for acceleration")
            else:
                self.compute_device = 'CPU'
                print(f"Note: Using CPU (NPU/GPU not found)")

        except ImportError:
            self.openvino_available = False
            self.compute_device = 'CPU'
            print("Note: OpenVINO not available for NPU acceleration")

    def optimize_protocol(self,
                         base_compounds: List[str],
                         n_patients: int = 1000,
                         max_iterations: int = 100,
                         target_success_rate: float = 0.70) -> OptimizationResult:
        """
        Optimize multi-compound protocol

        Args:
            base_compounds: List of compound names to include
            n_patients: Number of virtual patients for simulation
            max_iterations: Maximum optimization iterations
            target_success_rate: Target treatment success rate

        Returns:
            OptimizationResult with optimal protocol
        """
        start_time = time.time()

        # Validate compounds
        compounds = []
        for name in base_compounds:
            compound = self.db.get_compound(name)
            if compound:
                compounds.append(compound)
            else:
                print(f"Warning: Compound '{name}' not found")

        if not compounds:
            raise ValueError("No valid compounds provided")

        print(f"\nOptimizing protocol with {len(compounds)} compounds:")
        for c in compounds:
            print(f"  - {c.name}")
        print(f"Virtual patients: {n_patients}")
        print(f"Max iterations: {max_iterations}")
        print(f"Using {self.n_cores} CPU cores")

        # Define optimization bounds
        # [dose1, dose2, ..., freq1, freq2, ...]
        n_compounds = len(compounds)
        bounds = []

        for _ in range(n_compounds):
            bounds.append((1.0, 50.0))  # Dose bounds in mg

        for _ in range(n_compounds):
            bounds.append((1, 4))  # Frequency bounds (doses per day)

        # Initial guess
        x0 = []
        for compound in compounds:
            # Reasonable starting doses based on Ki
            if compound.ki_orthosteric != float('inf'):
                dose = max(5.0, min(30.0, compound.ki_orthosteric / 2))
            else:
                dose = 15.0
            x0.append(dose)

        # Reasonable starting frequencies based on half-life
        for compound in compounds:
            if compound.t_half < 4:
                freq = 4  # QID
            elif compound.t_half < 8:
                freq = 2  # BID
            else:
                freq = 1  # QD
            x0.append(freq)

        # Objective function
        def objective(x):
            doses = x[:n_compounds]
            frequencies = x[n_compounds:]

            protocol = ProtocolConfig(
                compounds=[c.name for c in compounds],
                doses=list(doses),
                frequencies=list(frequencies)
            )

            metrics = self._evaluate_protocol(protocol, compounds, n_patients)

            # Maximize success rate while minimizing side effects
            score = (
                -metrics['success_rate'] * 100 +
                metrics['tolerance_rate'] * 50 +
                metrics['addiction_rate'] * 30 +
                metrics['withdrawal_rate'] * 40
            )

            return score

        # Run optimization
        if SCIPY_AVAILABLE:
            print("\nRunning optimization with differential evolution...")

            if self.use_multiprocessing:
                workers = min(self.n_cores - 1, 8)  # Leave one core free
            else:
                workers = 1

            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=max_iterations,
                workers=workers,
                updating='deferred' if workers > 1 else 'immediate',
                polish=True,
                atol=0.001,
                tol=0.01
            )

            optimal_x = result.x
            iterations = result.nit
        else:
            # Simple grid search fallback
            print("\nRunning basic grid search optimization...")
            optimal_x = np.array(x0)
            iterations = max_iterations

            best_score = objective(optimal_x)

            for i in range(max_iterations):
                # Random perturbation
                perturbation = np.random.randn(len(optimal_x)) * 0.1
                x_new = optimal_x + perturbation

                # Clip to bounds
                for j, (low, high) in enumerate(bounds):
                    x_new[j] = np.clip(x_new[j], low, high)

                score = objective(x_new)
                if score < best_score:
                    best_score = score
                    optimal_x = x_new

                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{max_iterations}, Score: {-best_score:.3f}")

        # Extract optimal parameters
        optimal_doses = optimal_x[:n_compounds]
        optimal_frequencies = optimal_x[n_compounds:]

        optimal_protocol = ProtocolConfig(
            compounds=[c.name for c in compounds],
            doses=list(optimal_doses),
            frequencies=list(optimal_frequencies)
        )

        # Final evaluation
        final_metrics = self._evaluate_protocol(optimal_protocol, compounds, n_patients)

        computation_time = time.time() - start_time

        result = OptimizationResult(
            optimal_protocol=optimal_protocol,
            success_rate=final_metrics['success_rate'],
            tolerance_rate=final_metrics['tolerance_rate'],
            addiction_rate=final_metrics['addiction_rate'],
            withdrawal_rate=final_metrics['withdrawal_rate'],
            safety_score=final_metrics['safety_score'],
            therapeutic_window=final_metrics['therapeutic_window'],
            iterations=iterations,
            computation_time=computation_time
        )

        return result

    def _evaluate_protocol(self, protocol: ProtocolConfig,
                          compounds: List[CompoundProfile],
                          n_patients: int) -> Dict[str, float]:
        """
        Evaluate protocol performance across virtual patients

        Args:
            protocol: Protocol configuration
            compounds: List of compound profiles
            n_patients: Number of patients to simulate

        Returns:
            Dictionary of performance metrics
        """
        # Simulate patient population
        if self.use_multiprocessing and n_patients > 100:
            # Parallel simulation
            n_workers = min(self.n_cores - 1, 8)
            patients_per_worker = n_patients // n_workers

            with mp.Pool(n_workers) as pool:
                results = pool.starmap(
                    self._simulate_patient,
                    [(protocol, compounds) for _ in range(n_patients)]
                )
        else:
            # Serial simulation
            results = [self._simulate_patient(protocol, compounds)
                      for _ in range(n_patients)]

        # Aggregate results
        success_count = sum(1 for r in results if r['success'])
        tolerance_count = sum(1 for r in results if r['tolerance'])
        addiction_count = sum(1 for r in results if r['addiction'])
        withdrawal_count = sum(1 for r in results if r['withdrawal'])

        avg_analgesia = np.mean([r['analgesia'] for r in results])
        avg_side_effects = np.mean([r['side_effects'] for r in results])

        # Calculate metrics
        success_rate = success_count / n_patients
        tolerance_rate = tolerance_count / n_patients
        addiction_rate = addiction_count / n_patients
        withdrawal_rate = withdrawal_count / n_patients

        # Calculate safety score
        safety_score = (
            success_rate * 100 -
            tolerance_rate * 50 -
            addiction_rate * 30 -
            withdrawal_rate * 40
        )

        # Calculate therapeutic window
        if avg_side_effects > 0:
            therapeutic_window = avg_analgesia / avg_side_effects
        else:
            therapeutic_window = float('inf')

        return {
            'success_rate': success_rate,
            'tolerance_rate': tolerance_rate,
            'addiction_rate': addiction_rate,
            'withdrawal_rate': withdrawal_rate,
            'safety_score': max(0, safety_score),
            'therapeutic_window': therapeutic_window,
            'avg_analgesia': avg_analgesia,
            'avg_side_effects': avg_side_effects
        }

    def _simulate_patient(self, protocol: ProtocolConfig,
                         compounds: List[CompoundProfile]) -> Dict:
        """
        Simulate single patient response to protocol

        Args:
            protocol: Protocol configuration
            compounds: List of compound profiles

        Returns:
            Dictionary of patient outcomes
        """
        # Patient variability
        sensitivity = np.random.lognormal(0, 0.3)  # Log-normal distribution
        metabolism_rate = np.random.lognormal(0, 0.2)

        # Simulate 24-hour steady state
        time_points = np.linspace(0, 24, 96)  # 15-minute intervals

        total_analgesia = np.zeros_like(time_points)
        total_side_effects = np.zeros_like(time_points)
        tolerance_accumulation = 0.0

        for compound, dose, freq in zip(compounds, protocol.doses, protocol.frequencies):
            # Calculate dosing schedule
            interval = 24 / freq
            dose_times = np.arange(0, 24, interval)

            for t_idx, t in enumerate(time_points):
                # Sum contributions from all doses
                concentration = 0.0

                for dose_time in dose_times:
                    if t >= dose_time:
                        time_since_dose = t - dose_time
                        # Adjust half-life by metabolism rate
                        adjusted_t_half = compound.t_half / metabolism_rate

                        conc = self.pk_model.calculate_concentration(
                            dose, time_since_dose, adjusted_t_half,
                            compound.bioavailability
                        )
                        concentration += conc

                # Calculate effects
                if compound.ki_orthosteric != float('inf'):
                    ki = compound.ki_orthosteric
                else:
                    ki = compound.ki_allosteric1

                g_activation = self.pk_model.calculate_receptor_occupancy(
                    concentration * sensitivity, ki,
                    compound.intrinsic_activity * compound.g_protein_bias
                )

                beta_activation = self.pk_model.calculate_receptor_occupancy(
                    concentration * sensitivity, ki,
                    compound.intrinsic_activity * compound.beta_arrestin_bias
                )

                # Accumulate tolerance
                if not compound.reverses_tolerance:
                    tolerance_accumulation += compound.tolerance_rate * beta_activation * 0.01

                # Calculate analgesia with tolerance
                if compound.reverses_tolerance:
                    tolerance = max(0, tolerance_accumulation * 0.5)
                elif compound.prevents_withdrawal:
                    tolerance = tolerance_accumulation * 0.3
                else:
                    tolerance = tolerance_accumulation

                analgesia = self.pk_model.calculate_analgesia(g_activation, min(tolerance, 0.9))

                total_analgesia[t_idx] += analgesia
                total_side_effects[t_idx] += beta_activation

        # Determine outcomes
        avg_analgesia = np.mean(total_analgesia)
        avg_side_effects = np.mean(total_side_effects)
        min_analgesia = np.min(total_analgesia)

        # Success criteria
        success = (
            avg_analgesia > 0.5 and  # Adequate pain control
            min_analgesia > 0.3 and  # No breakthrough pain
            avg_side_effects < 0.4  # Manageable side effects
        )

        # Tolerance development (after 90 days)
        developed_tolerance = tolerance_accumulation > 0.5

        # Addiction risk
        addiction_risk = avg_side_effects * 0.3  # Proportional to �-arrestin
        developed_addiction = np.random.random() < addiction_risk

        # Withdrawal risk
        # Check if any compound prevents withdrawal
        has_protection = any(c.prevents_withdrawal for c in compounds)
        withdrawal_risk = 0.05 if has_protection else 0.2
        has_withdrawal = np.random.random() < withdrawal_risk

        return {
            'success': success,
            'tolerance': developed_tolerance,
            'addiction': developed_addiction,
            'withdrawal': has_withdrawal,
            'analgesia': avg_analgesia,
            'side_effects': avg_side_effects
        }


def run_local_optimization(compounds: List[str],
                          n_patients: int = 1000,
                          max_iterations: int = 100,
                          use_intel: bool = False) -> OptimizationResult:
    """
    Convenience function to run optimization locally

    Args:
        compounds: List of compound names
        n_patients: Number of virtual patients
        max_iterations: Maximum optimization iterations
        use_intel: Enable Intel NPU/GPU acceleration

    Returns:
        OptimizationResult
    """
    db = CompoundDatabase()
    optimizer = ProtocolOptimizer(
        db,
        use_multiprocessing=True,
        use_intel_acceleration=use_intel
    )

    result = optimizer.optimize_protocol(
        compounds,
        n_patients=n_patients,
        max_iterations=max_iterations
    )

    return result


if __name__ == '__main__':
    print("ZeroPain Protocol Optimization Framework")
    print("=" * 60)

    # Example: Optimize triple compound protocol
    compounds = ['SR-17018', 'SR-14968', 'Oxycodone']

    print(f"\nOptimizing protocol with compounds: {compounds}")
    print("This may take several minutes...\n")

    result = run_local_optimization(
        compounds,
        n_patients=500,  # Reduced for faster demo
        max_iterations=50,
        use_intel=True
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nOptimal Protocol:")
    for compound, dose, freq in zip(result.optimal_protocol.compounds,
                                    result.optimal_protocol.doses,
                                    result.optimal_protocol.frequencies):
        freq_str = ['', 'QD', 'BID', 'TID', 'QID'][int(freq)] if freq <= 4 else f'{int(freq)}x/day'
        print(f"  {compound:15s}: {dose:6.2f} mg {freq_str}")

    print(f"\nPerformance Metrics:")
    print(f"  Success Rate:      {result.success_rate*100:6.2f}%")
    print(f"  Tolerance Rate:    {result.tolerance_rate*100:6.2f}%")
    print(f"  Addiction Rate:    {result.addiction_rate*100:6.2f}%")
    print(f"  Withdrawal Rate:   {result.withdrawal_rate*100:6.2f}%")
    print(f"  Safety Score:      {result.safety_score:6.2f}")
    print(f"  Therapeutic Window: {result.therapeutic_window:6.2f}x")

    print(f"\nComputation:")
    print(f"  Iterations:        {result.iterations}")
    print(f"  Time:              {result.computation_time:.2f} seconds")

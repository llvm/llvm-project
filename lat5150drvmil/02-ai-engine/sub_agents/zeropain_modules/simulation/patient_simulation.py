#!/usr/bin/env python3
"""
ZeroPain Patient Simulation Framework
Large-scale patient population simulation with realistic variability
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial
import json
import time

from .opioid_analysis_tools import CompoundDatabase, CompoundProfile
from .opioid_optimization_framework import ProtocolConfig, PharmacokineticModel


@dataclass
class PatientProfile:
    """Individual patient characteristics"""
    patient_id: int
    age: int
    weight: float  # kg
    sex: str  # 'M' or 'F'
    metabolism_rate: float  # Multiplier for drug clearance
    sensitivity: float  # Receptor sensitivity multiplier
    pain_severity: float  # 0-10 baseline pain
    comorbidities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'age': self.age,
            'weight': self.weight,
            'sex': self.sex,
            'metabolism_rate': self.metabolism_rate,
            'sensitivity': self.sensitivity,
            'pain_severity': self.pain_severity,
            'comorbidities': self.comorbidities
        }


@dataclass
class SimulationResult:
    """Results from patient simulation"""
    patient: PatientProfile
    success: bool
    tolerance_developed: bool
    addiction_signs: bool
    withdrawal_symptoms: bool
    adverse_events: List[str]
    avg_pain_score: float
    avg_analgesia: float
    avg_side_effects: float
    quality_of_life: float

    def to_dict(self) -> Dict:
        return {
            'patient': self.patient.to_dict(),
            'success': self.success,
            'tolerance_developed': self.tolerance_developed,
            'addiction_signs': self.addiction_signs,
            'withdrawal_symptoms': self.withdrawal_symptoms,
            'adverse_events': self.adverse_events,
            'avg_pain_score': self.avg_pain_score,
            'avg_analgesia': self.avg_analgesia,
            'avg_side_effects': self.avg_side_effects,
            'quality_of_life': self.quality_of_life
        }


class PatientGenerator:
    """Generate realistic virtual patient populations"""

    @staticmethod
    def generate_patient(patient_id: int, seed: Optional[int] = None) -> PatientProfile:
        """Generate a single virtual patient with realistic characteristics"""
        if seed is not None:
            np.random.seed(seed + patient_id)

        # Age distribution (18-85)
        age = int(np.random.beta(2, 3) * 67 + 18)

        # Weight distribution (45-120 kg)
        sex = np.random.choice(['M', 'F'])
        if sex == 'M':
            weight = np.random.normal(82, 15)
        else:
            weight = np.random.normal(70, 13)
        weight = np.clip(weight, 45, 120)

        # Metabolism rate (log-normal distribution)
        # Affected by age, genetics, comorbidities
        base_metabolism = np.random.lognormal(0, 0.25)

        # Age effect on metabolism
        if age > 65:
            base_metabolism *= 0.8
        elif age < 25:
            base_metabolism *= 1.2

        # Receptor sensitivity (log-normal distribution)
        sensitivity = np.random.lognormal(0, 0.3)

        # Pain severity (beta distribution)
        pain_severity = np.random.beta(3, 2) * 10

        # Comorbidities (more common with age)
        comorbidities = []
        comorbidity_prob = 0.1 + (age - 18) / 200

        possible_comorbidities = [
            'hypertension', 'diabetes', 'depression', 'anxiety',
            'arthritis', 'COPD', 'kidney_disease', 'liver_disease'
        ]

        for condition in possible_comorbidities:
            if np.random.random() < comorbidity_prob:
                comorbidities.append(condition)

        # Liver disease reduces metabolism
        if 'liver_disease' in comorbidities:
            base_metabolism *= 0.6

        # Kidney disease affects clearance
        if 'kidney_disease' in comorbidities:
            base_metabolism *= 0.7

        return PatientProfile(
            patient_id=patient_id,
            age=age,
            weight=weight,
            sex=sex,
            metabolism_rate=base_metabolism,
            sensitivity=sensitivity,
            pain_severity=pain_severity,
            comorbidities=comorbidities
        )

    @classmethod
    def generate_population(cls, n_patients: int,
                          seed: Optional[int] = 42) -> List[PatientProfile]:
        """Generate population of virtual patients"""
        return [cls.generate_patient(i, seed) for i in range(n_patients)]


class PatientSimulator:
    """Simulate patient response to opioid protocols"""

    def __init__(self, compound_database: CompoundDatabase):
        self.db = compound_database
        self.pk_model = PharmacokineticModel()

    def simulate_patient(self, patient: PatientProfile,
                        protocol: ProtocolConfig,
                        duration_days: int = 90) -> SimulationResult:
        """
        Simulate patient response to protocol over time

        Args:
            patient: Patient profile
            protocol: Treatment protocol
            duration_days: Simulation duration in days

        Returns:
            SimulationResult with outcomes
        """
        # Get compound profiles
        compounds = [self.db.get_compound(name) for name in protocol.compounds]
        compounds = [c for c in compounds if c is not None]

        if not compounds:
            raise ValueError("No valid compounds in protocol")

        # Simulation parameters
        hours_per_day = 24
        time_step = 0.25  # 15-minute intervals
        n_timepoints = int(duration_days * hours_per_day / time_step)

        # Initialize tracking arrays
        pain_scores = np.zeros(n_timepoints)
        analgesia_levels = np.zeros(n_timepoints)
        side_effect_levels = np.zeros(n_timepoints)
        tolerance_level = 0.0
        tolerance_history = []

        # Adverse events tracking
        adverse_events = []

        # Volume of distribution adjusted for weight
        volume_dist = patient.weight * 0.7  # L/kg

        # Simulate each timepoint
        for t_idx in range(n_timepoints):
            t_hours = t_idx * time_step
            t_days = t_hours / hours_per_day

            # Calculate time of day for dosing
            time_of_day = t_hours % hours_per_day

            total_analgesia = 0.0
            total_side_effects = 0.0

            # Calculate contribution from each compound
            for compound, dose, freq in zip(compounds, protocol.doses, protocol.frequencies):
                # Determine if dose should be administered
                interval = hours_per_day / freq
                dose_times = [i * interval for i in range(int(freq))]

                # Find time since last dose
                time_since_dose = min([time_of_day - dt if time_of_day >= dt
                                      else time_of_day + hours_per_day - dt
                                      for dt in dose_times])

                # Calculate concentration
                adjusted_t_half = compound.t_half / patient.metabolism_rate

                concentration = self.pk_model.calculate_concentration(
                    dose, time_since_dose, adjusted_t_half,
                    compound.bioavailability, volume_dist
                )

                # Determine binding site
                if compound.ki_orthosteric != float('inf'):
                    ki = compound.ki_orthosteric
                elif compound.ki_allosteric1 != float('inf'):
                    ki = compound.ki_allosteric1
                else:
                    ki = 50.0  # Default

                # Calculate receptor occupancy with patient sensitivity
                g_activation = self.pk_model.calculate_receptor_occupancy(
                    concentration * patient.sensitivity,
                    ki,
                    compound.intrinsic_activity * compound.g_protein_bias
                )

                beta_activation = self.pk_model.calculate_receptor_occupancy(
                    concentration * patient.sensitivity,
                    ki,
                    compound.intrinsic_activity * compound.beta_arrestin_bias
                )

                # Update tolerance
                if not compound.reverses_tolerance:
                    tolerance_increment = compound.tolerance_rate * beta_activation * 0.0001
                    tolerance_level += tolerance_increment

                # Apply tolerance reversal effects
                if compound.reverses_tolerance and tolerance_level > 0:
                    tolerance_level *= 0.9995  # Gradual reversal

                # Calculate effective tolerance
                if compound.reverses_tolerance:
                    effective_tolerance = max(0, tolerance_level * 0.3)
                elif compound.prevents_withdrawal:
                    effective_tolerance = tolerance_level * 0.5
                else:
                    effective_tolerance = tolerance_level

                effective_tolerance = min(effective_tolerance, 0.95)

                # Calculate analgesia
                analgesia = self.pk_model.calculate_analgesia(
                    g_activation, effective_tolerance
                )

                total_analgesia += analgesia
                total_side_effects += beta_activation

            # Cap maximum effects
            total_analgesia = min(total_analgesia, 1.0)
            total_side_effects = min(total_side_effects, 1.0)

            # Calculate pain score (0-10 scale)
            baseline_pain = patient.pain_severity
            pain_relief = total_analgesia * baseline_pain
            current_pain = max(0, baseline_pain - pain_relief)

            # Check for adverse events
            if total_side_effects > 0.7:
                if np.random.random() < 0.001:  # Low probability per timepoint
                    adverse_events.append(f"High side effects at day {t_days:.1f}")

            # Store results
            pain_scores[t_idx] = current_pain
            analgesia_levels[t_idx] = total_analgesia
            side_effect_levels[t_idx] = total_side_effects
            tolerance_history.append(tolerance_level)

        # Calculate outcomes
        avg_pain_score = np.mean(pain_scores)
        avg_analgesia = np.mean(analgesia_levels)
        avg_side_effects = np.mean(side_effect_levels)
        final_tolerance = tolerance_level

        # Success criteria
        success = (
            avg_pain_score < 4.0 and  # Adequate pain control
            avg_analgesia > 0.5 and
            avg_side_effects < 0.4 and
            final_tolerance < 0.5
        )

        # Tolerance development
        tolerance_developed = final_tolerance > 0.5

        # Addiction risk factors
        addiction_risk = avg_side_effects * 0.25
        if 'depression' in patient.comorbidities:
            addiction_risk *= 1.5
        if 'anxiety' in patient.comorbidities:
            addiction_risk *= 1.3

        addiction_signs = np.random.random() < addiction_risk

        # Withdrawal assessment
        has_withdrawal_protection = any(
            self.db.get_compound(name).prevents_withdrawal
            for name in protocol.compounds
            if self.db.get_compound(name)
        )

        if has_withdrawal_protection:
            withdrawal_risk = 0.02
        else:
            withdrawal_risk = 0.15 + final_tolerance * 0.1

        withdrawal_symptoms = np.random.random() < withdrawal_risk

        # Quality of life score (0-1)
        pain_impact = (10 - avg_pain_score) / 10
        side_effect_impact = 1 - avg_side_effects
        quality_of_life = (pain_impact * 0.6 + side_effect_impact * 0.4)

        # Additional adverse events
        if avg_side_effects > 0.6:
            adverse_events.append("Persistent side effects")
        if final_tolerance > 0.7:
            adverse_events.append("Significant tolerance development")

        return SimulationResult(
            patient=patient,
            success=success,
            tolerance_developed=tolerance_developed,
            addiction_signs=addiction_signs,
            withdrawal_symptoms=withdrawal_symptoms,
            adverse_events=adverse_events,
            avg_pain_score=avg_pain_score,
            avg_analgesia=avg_analgesia,
            avg_side_effects=avg_side_effects,
            quality_of_life=quality_of_life
        )


class PopulationSimulation:
    """Large-scale population simulation"""

    def __init__(self, compound_database: CompoundDatabase,
                 use_multiprocessing: bool = True):
        self.db = compound_database
        self.simulator = PatientSimulator(compound_database)
        self.use_multiprocessing = use_multiprocessing
        self.n_cores = mp.cpu_count()

    def run_simulation(self, protocol: ProtocolConfig,
                      n_patients: int = 100000,
                      duration_days: int = 90,
                      seed: int = 42) -> Dict:
        """
        Run large-scale population simulation

        Args:
            protocol: Treatment protocol
            n_patients: Number of virtual patients
            duration_days: Simulation duration
            seed: Random seed

        Returns:
            Dictionary with aggregated results
        """
        print(f"\nGenerating {n_patients:,} virtual patients...")
        start_time = time.time()

        # Generate patient population
        generator = PatientGenerator()
        patients = generator.generate_population(n_patients, seed)

        gen_time = time.time() - start_time
        print(f"  Generated in {gen_time:.2f} seconds")

        print(f"\nSimulating {duration_days}-day protocol...")
        print(f"  Compounds: {', '.join(protocol.compounds)}")
        print(f"  Using {self.n_cores} CPU cores")

        sim_start = time.time()

        # Run simulations
        if self.use_multiprocessing and n_patients > 100:
            # Parallel simulation
            n_workers = min(self.n_cores - 1, 16)
            print(f"  Parallel processing with {n_workers} workers")

            with mp.Pool(n_workers) as pool:
                simulate_func = partial(
                    self._simulate_wrapper,
                    protocol=protocol,
                    duration_days=duration_days
                )
                results = pool.map(simulate_func, patients)
        else:
            # Serial simulation
            print(f"  Serial processing")
            results = [
                self.simulator.simulate_patient(patient, protocol, duration_days)
                for patient in patients
            ]

        sim_time = time.time() - sim_start
        print(f"  Simulated in {sim_time:.2f} seconds")
        print(f"  Rate: {n_patients/sim_time:.0f} patients/second")

        # Aggregate results
        print("\nAggregating results...")
        aggregated = self._aggregate_results(results)

        total_time = time.time() - start_time
        aggregated['computation_time'] = total_time
        aggregated['n_patients'] = n_patients

        return aggregated

    def _simulate_wrapper(self, patient: PatientProfile,
                         protocol: ProtocolConfig,
                         duration_days: int) -> SimulationResult:
        """Wrapper for multiprocessing"""
        return self.simulator.simulate_patient(patient, protocol, duration_days)

    def _aggregate_results(self, results: List[SimulationResult]) -> Dict:
        """Aggregate simulation results"""
        n_patients = len(results)

        # Count outcomes
        success_count = sum(1 for r in results if r.success)
        tolerance_count = sum(1 for r in results if r.tolerance_developed)
        addiction_count = sum(1 for r in results if r.addiction_signs)
        withdrawal_count = sum(1 for r in results if r.withdrawal_symptoms)
        adverse_count = sum(1 for r in results if len(r.adverse_events) > 0)

        # Calculate metrics
        metrics = {
            'success_rate': success_count / n_patients,
            'tolerance_rate': tolerance_count / n_patients,
            'addiction_rate': addiction_count / n_patients,
            'withdrawal_rate': withdrawal_count / n_patients,
            'adverse_event_rate': adverse_count / n_patients,
            'avg_pain_score': np.mean([r.avg_pain_score for r in results]),
            'avg_analgesia': np.mean([r.avg_analgesia for r in results]),
            'avg_side_effects': np.mean([r.avg_side_effects for r in results]),
            'avg_quality_of_life': np.mean([r.quality_of_life for r in results]),
        }

        # Distribution statistics
        metrics['pain_score_std'] = np.std([r.avg_pain_score for r in results])
        metrics['analgesia_std'] = np.std([r.avg_analgesia for r in results])
        metrics['qol_std'] = np.std([r.quality_of_life for r in results])

        return metrics


def run_100k_simulation(protocol: ProtocolConfig) -> Dict:
    """Convenience function to run 100k patient simulation"""
    db = CompoundDatabase()
    simulation = PopulationSimulation(db, use_multiprocessing=True)

    results = simulation.run_simulation(
        protocol,
        n_patients=100000,
        duration_days=90
    )

    return results


if __name__ == '__main__':
    print("ZeroPain Patient Simulation Framework")
    print("=" * 60)

    # Example protocol
    protocol = ProtocolConfig(
        compounds=['SR-17018', 'SR-14968', 'Oxycodone'],
        doses=[16.17, 25.31, 5.07],
        frequencies=[2, 1, 4]
    )

    # Run simulation (smaller for demo)
    db = CompoundDatabase()
    simulation = PopulationSimulation(db, use_multiprocessing=True)

    results = simulation.run_simulation(
        protocol,
        n_patients=10000,  # Reduced for demo
        duration_days=90
    )

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nPrimary Outcomes (n={results['n_patients']:,}):")
    print(f"  Treatment Success:     {results['success_rate']*100:6.2f}%")
    print(f"  Tolerance Development: {results['tolerance_rate']*100:6.2f}%")
    print(f"  Addiction Signs:       {results['addiction_rate']*100:6.2f}%")
    print(f"  Withdrawal Symptoms:   {results['withdrawal_rate']*100:6.2f}%")
    print(f"  Adverse Events:        {results['adverse_event_rate']*100:6.2f}%")

    print(f"\nClinical Metrics:")
    print(f"  Average Pain Score:    {results['avg_pain_score']:6.2f} / 10")
    print(f"  Average Analgesia:     {results['avg_analgesia']*100:6.2f}%")
    print(f"  Average Side Effects:  {results['avg_side_effects']*100:6.2f}%")
    print(f"  Quality of Life:       {results['avg_quality_of_life']*100:6.2f}%")

    print(f"\nComputation Time:        {results['computation_time']:.2f} seconds")

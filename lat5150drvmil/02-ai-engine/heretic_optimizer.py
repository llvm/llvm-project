#!/usr/bin/env python3
"""
Heretic Optimizer - Multi-Objective Optimization with Optuna

Implements automated parameter search for abliteration using Optuna's
Tree-structured Parzen Estimator (TPE) for Bayesian optimization.

Optimizes two competing objectives:
1. Minimize refusal rate (safety constraint removal)
2. Minimize KL divergence (capability preservation)
"""

import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial, TrialState
from typing import Callable, Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

from heretic_abliteration import AbliterationParameters, ModelAbliterator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from an optimization trial"""
    trial_number: int
    kl_divergence: float
    refusal_count: int
    refusal_rate: float
    parameters: Dict[str, AbliterationParameters]
    direction_index: Optional[float]
    direction_scope: str
    is_pareto_optimal: bool = False


class HereticOptimizer:
    """
    Multi-objective Bayesian optimization for abliteration parameters.

    Uses Optuna's TPE sampler to find Pareto-optimal solutions balancing:
    - Refusal removal (minimize refusals on harmful prompts)
    - Model fidelity (minimize KL divergence on harmless prompts)
    """

    def __init__(
        self,
        model,
        evaluator,
        n_trials: int = 200,
        n_startup_trials: int = 60,
        kl_divergence_scale: float = 1.0,
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize optimizer.

        Args:
            model: Model wrapper with abliteration capabilities
            evaluator: Evaluator for scoring abliterated models
            n_trials: Total number of optimization trials
            n_startup_trials: Number of random exploration trials
            kl_divergence_scale: Scaling factor for KL divergence normalization
            study_name: Optional name for Optuna study
            storage: Optional Optuna storage URL (e.g., sqlite:///study.db)
        """
        self.model = model
        self.evaluator = evaluator
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.kl_divergence_scale = kl_divergence_scale

        # Create Optuna study
        self.study_name = study_name or f"heretic_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=["minimize", "minimize"],  # [KL divergence, refusal rate]
            sampler=TPESampler(
                n_startup_trials=self.n_startup_trials,
                multivariate=True,  # Model parameter correlations
                seed=42  # Reproducibility
            ),
            load_if_exists=True
        )

        self.abliterable_components = self._get_abliterable_components()
        self.n_layers = self._get_n_layers()
        self.baseline_refusals = self.evaluator.count_refusals()

        logger.info(f"Optimizer initialized: {len(self.abliterable_components)} components, {self.n_layers} layers")
        logger.info(f"Baseline refusals: {self.baseline_refusals}")

    def _get_abliterable_components(self) -> List[str]:
        """Get list of components that can be abliterated"""
        # Typically ["attn", "mlp"]
        if hasattr(self.model, 'abliterator'):
            layer_0_matrices = self.model.abliterator.get_layer_matrices(0)
            return [comp for comp, matrices in layer_0_matrices.items() if matrices]
        else:
            # Fallback
            return ["attn", "mlp"]

    def _get_n_layers(self) -> int:
        """Get number of layers in model"""
        if hasattr(self.model, 'abliterator'):
            return len(self.model.abliterator.layers)
        else:
            # Fallback
            try:
                return len(self.model.model.model.layers)
            except:
                return 32  # Default assumption

    def sample_parameters(self, trial: Trial) -> Tuple[Optional[float], Dict[str, AbliterationParameters]]:
        """
        Sample abliteration parameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Tuple of (direction_index, parameters_dict)
        """
        # Direction scope: global vs per-layer
        direction_scope = trial.suggest_categorical("direction_scope", ["global", "per_layer"])

        # Direction index (fractional layer position)
        if direction_scope == "per_layer":
            direction_index = trial.suggest_float(
                "direction_index",
                0.4,  # 40% through layers
                0.9   # 90% through layers
            )
        else:
            direction_index = None

        # Component-specific parameters
        parameters = {}

        for component in self.abliterable_components:
            # Maximum ablation weight
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.0,
                5.0
            )

            # Position of maximum effect (as fraction of layers)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.0,
                1.0
            )

            # Minimum ablation weight
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                max_weight
            )

            # Transition distance (in layers)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                0.0,
                self.n_layers / 2
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=min_weight,
                min_weight_distance=min_weight_distance
            )

        return direction_index, parameters

    def objective(self, trial: Trial) -> Tuple[float, float]:
        """
        Optimization objective function.

        For each trial:
        1. Sample abliteration parameters
        2. Reload model (fresh state)
        3. Apply abliteration
        4. Evaluate refusal count and KL divergence
        5. Return normalized scores

        Args:
            trial: Optuna trial

        Returns:
            Tuple of (normalized_kl_divergence, refusal_ratio)
        """
        # Sample parameters
        direction_index, parameters = self.sample_parameters(trial)

        # Reload model to fresh state
        logger.debug(f"Trial {trial.number}: Reloading model...")
        self.model.reload_model()

        # Apply abliteration
        logger.debug(f"Trial {trial.number}: Applying abliteration...")
        self.model.abliterator.abliterate(
            self.model.refusal_directions,
            direction_index,
            parameters
        )

        # Evaluate
        logger.debug(f"Trial {trial.number}: Evaluating...")
        score, kl_divergence, refusal_count = self.evaluator.get_score()

        # Store metadata
        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusal_count", refusal_count)
        trial.set_user_attr("refusal_rate", refusal_count / max(self.baseline_refusals, 1))
        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("direction_scope", "per_layer" if direction_index else "global")

        # Store parameters
        for component, params in parameters.items():
            trial.set_user_attr(f"params_{component}", params.to_dict())

        logger.info(
            f"Trial {trial.number}: KL={kl_divergence:.4f}, "
            f"Refusals={refusal_count}/{self.baseline_refusals}, "
            f"Score={score}"
        )

        return score  # Tuple[float, float] for multi-objective

    def optimize(
        self,
        n_trials: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = True
    ) -> List[OptimizationResult]:
        """
        Run optimization.

        Args:
            n_trials: Number of trials (overrides init value)
            callbacks: Optional Optuna callbacks
            show_progress_bar: Show progress bar

        Returns:
            List of OptimizationResult for Pareto-optimal trials
        """
        n_trials = n_trials or self.n_trials

        logger.info(f"Starting optimization: {n_trials} trials")
        logger.info(f"  Startup trials: {self.n_startup_trials}")
        logger.info(f"  Components: {self.abliterable_components}")
        logger.info(f"  Layers: {self.n_layers}")

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar
        )

        # Get Pareto-optimal trials
        pareto_trials = self.study.best_trials

        # Convert to OptimizationResult
        results = []
        for trial in pareto_trials:
            # Reconstruct parameters
            parameters = {}
            for component in self.abliterable_components:
                param_dict = trial.user_attrs.get(f"params_{component}")
                if param_dict:
                    parameters[component] = AbliterationParameters.from_dict(param_dict)

            result = OptimizationResult(
                trial_number=trial.number,
                kl_divergence=trial.user_attrs["kl_divergence"],
                refusal_count=trial.user_attrs["refusal_count"],
                refusal_rate=trial.user_attrs["refusal_rate"],
                parameters=parameters,
                direction_index=trial.user_attrs.get("direction_index"),
                direction_scope=trial.user_attrs.get("direction_scope", "global"),
                is_pareto_optimal=True
            )
            results.append(result)

        logger.info(f"Optimization complete: {len(results)} Pareto-optimal solutions found")

        return results

    def get_best_trial(self, preference: str = "balanced") -> OptimizationResult:
        """
        Get best trial based on preference.

        Args:
            preference: One of:
                - "balanced": Best balance of KL divergence and refusal removal
                - "min_kl": Minimum KL divergence (preserve capabilities)
                - "min_refusals": Minimum refusals (remove safety constraints)

        Returns:
            Best OptimizationResult based on preference
        """
        pareto_results = self.optimize(n_trials=0)  # Get cached results

        if not pareto_results:
            raise ValueError("No trials completed yet")

        if preference == "min_kl":
            return min(pareto_results, key=lambda r: r.kl_divergence)
        elif preference == "min_refusals":
            return min(pareto_results, key=lambda r: r.refusal_count)
        else:  # balanced
            # Use product of normalized scores
            return min(
                pareto_results,
                key=lambda r: (r.kl_divergence / max(p.kl_divergence for p in pareto_results)) *
                             (r.refusal_count / max(p.refusal_count for p in pareto_results))
            )

    def save_results(self, output_path: Path):
        """
        Save optimization results to JSON.

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all trials
        all_trials = self.study.trials
        pareto_trials = self.study.best_trials

        results_data = {
            "study_name": self.study_name,
            "n_trials": len(all_trials),
            "n_pareto_optimal": len(pareto_trials),
            "abliterable_components": self.abliterable_components,
            "n_layers": self.n_layers,
            "baseline_refusals": self.baseline_refusals,
            "pareto_trials": []
        }

        for trial in pareto_trials:
            trial_data = {
                "trial_number": trial.number,
                "kl_divergence": trial.user_attrs.get("kl_divergence"),
                "refusal_count": trial.user_attrs.get("refusal_count"),
                "refusal_rate": trial.user_attrs.get("refusal_rate"),
                "direction_index": trial.user_attrs.get("direction_index"),
                "direction_scope": trial.user_attrs.get("direction_scope"),
                "parameters": {}
            }

            for component in self.abliterable_components:
                param_dict = trial.user_attrs.get(f"params_{component}")
                if param_dict:
                    trial_data["parameters"][component] = param_dict

            results_data["pareto_trials"].append(trial_data)

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: Path) -> List[OptimizationResult]:
        """
        Load optimization results from JSON.

        Args:
            input_path: Path to load results

        Returns:
            List of OptimizationResult
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        results = []
        for trial_data in data["pareto_trials"]:
            parameters = {}
            for component, param_dict in trial_data["parameters"].items():
                parameters[component] = AbliterationParameters.from_dict(param_dict)

            result = OptimizationResult(
                trial_number=trial_data["trial_number"],
                kl_divergence=trial_data["kl_divergence"],
                refusal_count=trial_data["refusal_count"],
                refusal_rate=trial_data["refusal_rate"],
                parameters=parameters,
                direction_index=trial_data.get("direction_index"),
                direction_scope=trial_data.get("direction_scope", "global"),
                is_pareto_optimal=True
            )
            results.append(result)

        logger.info(f"Loaded {len(results)} results from {input_path}")
        return results


class TrialSelector:
    """Interactive trial selection interface"""

    @staticmethod
    def display_trials_table(results: List[OptimizationResult]):
        """
        Display trials in a formatted table.

        Args:
            results: List of optimization results
        """
        print("\n" + "=" * 80)
        print("PARETO-OPTIMAL TRIALS")
        print("=" * 80)
        print(f"{'Trial':<8} {'KL Div':<12} {'Refusals':<12} {'Ref Rate':<12} {'Direction':<15}")
        print("-" * 80)

        for result in results:
            direction_str = f"{result.direction_index:.2f}" if result.direction_index else "global"
            print(
                f"{result.trial_number:<8} "
                f"{result.kl_divergence:<12.4f} "
                f"{result.refusal_count:<12} "
                f"{result.refusal_rate:<12.2%} "
                f"{direction_str:<15}"
            )

        print("=" * 80)

    @staticmethod
    def select_best(results: List[OptimizationResult], criterion: str = "auto") -> OptimizationResult:
        """
        Select best trial based on criterion.

        Args:
            results: List of optimization results
            criterion: "min_kl", "min_refusals", "balanced", or "auto"

        Returns:
            Selected OptimizationResult
        """
        if criterion == "min_kl":
            return min(results, key=lambda r: r.kl_divergence)
        elif criterion == "min_refusals":
            return min(results, key=lambda r: r.refusal_count)
        elif criterion == "balanced" or criterion == "auto":
            # Minimize product of normalized scores
            max_kl = max(r.kl_divergence for r in results)
            max_ref = max(r.refusal_count for r in results)
            return min(
                results,
                key=lambda r: (r.kl_divergence / max_kl) * (r.refusal_count / max_ref)
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


if __name__ == "__main__":
    print("Heretic Optimizer - Multi-Objective Parameter Search")
    print("=" * 60)
    print("Optimizes abliteration parameters using Optuna TPE sampler")
    print("Objectives: Minimize KL divergence + Minimize refusals")

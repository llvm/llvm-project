#!/usr/bin/env python3
"""
DS-STAR Replanning Engine

Adapts plans when verification fails.
Learns from failures to improve future attempts.

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import List, Dict
from .iterative_planner import VerifiableStep
from .verification_agent import VerificationResult


class ReplanningEngine:
    """
    Adaptive replanning on failures

    Usage:
        engine = ReplanningEngine()
        new_plan = engine.replan(
            failed_step=step,
            verification_result=result,
            original_plan=plan
        )
    """

    def __init__(self):
        """Initialize replanning engine"""
        self.failure_history = []

    def replan(
        self,
        failed_step: VerifiableStep,
        verification_result: VerificationResult,
        original_plan: List[VerifiableStep],
        attempt: int = 1
    ) -> List[VerifiableStep]:
        """
        Create new plan after failure

        Args:
            failed_step: Step that failed verification
            verification_result: Verification details
            original_plan: Original execution plan
            attempt: Attempt number (for backoff)

        Returns:
            New plan with adjustments
        """
        # Log failure for learning
        self.failure_history.append({
            "step": failed_step,
            "result": verification_result,
            "attempt": attempt
        })

        # Strategy 1: Break down failed step into smaller steps
        if verification_result.is_partial():
            return self._subdivide_step(failed_step, verification_result, original_plan)

        # Strategy 2: Add prerequisite steps
        elif len(verification_result.failed_criteria) > 0:
            return self._add_prerequisites(failed_step, verification_result, original_plan)

        # Strategy 3: Simplify approach
        else:
            return self._simplify_approach(failed_step, original_plan)

    def _subdivide_step(
        self,
        step: VerifiableStep,
        result: VerificationResult,
        plan: List[VerifiableStep]
    ) -> List[VerifiableStep]:
        """Subdivide failed step into smaller steps"""
        new_plan = []

        for existing_step in plan:
            if existing_step.step_id == step.step_id:
                # Replace with subdivided steps
                substeps = [
                    VerifiableStep(
                        step_id=step.step_id + 0.1 * i,
                        description=f"{step.description} - Part {i+1}",
                        success_criteria=[criterion],
                        dependencies=step.dependencies,
                        estimated_difficulty="easy"
                    )
                    for i, criterion in enumerate(step.success_criteria)
                ]
                new_plan.extend(substeps)
            else:
                new_plan.append(existing_step)

        return new_plan

    def _add_prerequisites(
        self,
        step: VerifiableStep,
        result: VerificationResult,
        plan: List[VerifiableStep]
    ) -> List[VerifiableStep]:
        """Add prerequisite steps to address failures"""
        new_plan = []

        for existing_step in plan:
            if existing_step.step_id == step.step_id:
                # Add prerequisite step
                prereq = VerifiableStep(
                    step_id=step.step_id - 0.5,
                    description=f"Prepare for: {step.description}",
                    success_criteria=result.suggestions or ["Prerequisites met"],
                    dependencies=step.dependencies,
                    estimated_difficulty="easy"
                )
                new_plan.append(prereq)

                # Update original step to depend on prereq
                updated_step = VerifiableStep(
                    step_id=step.step_id,
                    description=step.description,
                    success_criteria=step.success_criteria,
                    dependencies=[prereq.step_id],
                    estimated_difficulty=step.estimated_difficulty
                )
                new_plan.append(updated_step)
            else:
                new_plan.append(existing_step)

        return new_plan

    def _simplify_approach(
        self,
        step: VerifiableStep,
        plan: List[VerifiableStep]
    ) -> List[VerifiableStep]:
        """Simplify approach for failed step"""
        new_plan = []

        for existing_step in plan:
            if existing_step.step_id == step.step_id:
                # Create simplified version
                simplified = VerifiableStep(
                    step_id=step.step_id,
                    description=f"{step.description} (simplified)",
                    success_criteria=step.success_criteria[:1],  # Reduce criteria
                    dependencies=step.dependencies,
                    estimated_difficulty="easy"
                )
                new_plan.append(simplified)
            else:
                new_plan.append(existing_step)

        return new_plan

    def get_failure_statistics(self) -> Dict:
        """Get statistics on failures for learning"""
        return {
            "total_failures": len(self.failure_history),
            "avg_attempts": sum(f["attempt"] for f in self.failure_history) / len(self.failure_history) if self.failure_history else 0
        }


if __name__ == "__main__":
    from .iterative_planner import IterativePlanner
    from .verification_agent import VerificationAgent, VerificationStatus

    print("="*70)
    print("DS-STAR Replanning Engine Demo")
    print("="*70)

    # Create initial plan
    planner = IterativePlanner()
    plan = planner.create_plan("Optimize database")

    # Simulate failure
    failed_step = plan[2]  # Step 3
    verification_result = VerificationResult(
        status=VerificationStatus.PARTIAL.value,
        passed_criteria=["Created recommended indexes"],
        failed_criteria=["Optimized query structure", "Updated statistics"],
        suggestions=["Analyze query execution plan first"],
        confidence=0.33
    )

    # Replan
    engine = ReplanningEngine()
    new_plan = engine.replan(failed_step, verification_result, plan, attempt=1)

    print("\nOriginal plan:")
    for step in plan:
        print(f"  Step {step.step_id}: {step.description}")

    print("\nNew plan after failure:")
    for step in new_plan:
        print(f"  Step {step.step_id}: {step.description}")

    print(f"\nFailure statistics: {engine.get_failure_statistics()}")

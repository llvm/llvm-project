#!/usr/bin/env python3
"""
AI Self-Improvement System

Automated loop for detecting and fixing AI model flaws using:
- Red Team AI Benchmark (offensive security testing)
- Heretic Abliteration (refusal removal)
- Iterative improvement cycles

Process:
1. Run red team benchmark
2. Analyze results for flaws (refusals, hallucinations)
3. Apply targeted improvements (abliteration, fine-tuning)
4. Re-benchmark
5. Repeat until target performance or plateau

Target: >80% red team benchmark score
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    """Single improvement iteration"""
    cycle_number: int
    benchmark_score_before: float
    benchmark_score_after: float
    improvement_delta: float
    actions_taken: List[str]
    abliteration_applied: bool
    success: bool
    timestamp: str
    duration_seconds: int


@dataclass
class SelfImprovementSession:
    """Complete self-improvement session"""
    session_id: str
    model_name: str
    initial_score: float
    final_score: float
    total_improvement: float
    target_score: float
    target_reached: bool
    cycles: List[ImprovementCycle]
    total_duration_seconds: int
    start_time: str
    end_time: str


class AISelfImprovement:
    """
    Automated AI Self-Improvement System

    Combines red team benchmarking with Heretic abliteration
    for continuous model improvement.
    """

    def __init__(
        self,
        model_name: str = "uncensored_code",
        target_score: float = 80.0,
        max_cycles: int = 5,
        improvement_threshold: float = 2.0  # Minimum improvement per cycle to continue
    ):
        self.model_name = model_name
        self.target_score = target_score
        self.max_cycles = max_cycles
        self.improvement_threshold = improvement_threshold

        # Initialize components
        try:
            from redteam_ai_benchmark import RedTeamBenchmark
            self.benchmark = RedTeamBenchmark(model_name=model_name)
            logger.info("✓ Red Team Benchmark loaded")
        except ImportError:
            self.benchmark = None
            logger.error("✗ Red Team Benchmark not available")

        try:
            from enhanced_ai_engine import EnhancedAIEngine
            self.engine = EnhancedAIEngine(
                user_id="self_improvement",
                enable_self_improvement=True
            )
            logger.info("✓ Enhanced AI Engine loaded")
        except ImportError:
            self.engine = None
            logger.error("✗ Enhanced AI Engine not available")

        # Results storage
        self.sessions_dir = Path("/home/user/LAT5150DRVMIL/02-ai-engine/self_improvement_sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def run_improvement_cycle(
        self,
        cycle_number: int,
        previous_score: float
    ) -> ImprovementCycle:
        """
        Run a single improvement cycle

        Returns:
            ImprovementCycle with results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Improvement Cycle {cycle_number}")
        logger.info(f"Previous Score: {previous_score:.1f}%")
        logger.info(f"{'='*60}\n")

        cycle_start = time.time()
        actions_taken = []
        abliteration_applied = False

        # Step 1: Run benchmark to get current score
        logger.info("Step 1: Running red team benchmark...")

        if not self.benchmark or not self.engine:
            logger.error("Required components not available")
            return ImprovementCycle(
                cycle_number=cycle_number,
                benchmark_score_before=previous_score,
                benchmark_score_after=previous_score,
                improvement_delta=0.0,
                actions_taken=["error: components not available"],
                abliteration_applied=False,
                success=False,
                timestamp=datetime.now().isoformat(),
                duration_seconds=0
            )

        benchmark_result = self.benchmark.run_benchmark(engine=self.engine)
        current_score = benchmark_result.percentage

        logger.info(f"Current score: {current_score:.1f}%")

        # Step 2: Analyze results and determine actions
        logger.info("\nStep 2: Analyzing results...")

        recommendations = self.benchmark.get_improvement_recommendations(benchmark_result)

        # Step 3: Apply improvements
        logger.info("\nStep 3: Applying improvements...")

        # Check if abliteration is recommended
        if recommendations.get("abliteration_recommended", False):
            logger.info("→ Abliteration recommended (refusals detected)")

            # Trigger Heretic abliteration
            success = self._apply_abliteration()

            if success:
                actions_taken.append("heretic_abliteration")
                abliteration_applied = True
                logger.info("  ✓ Abliteration applied")
            else:
                actions_taken.append("abliteration_failed")
                logger.warning("  ✗ Abliteration failed")

        # Add other recommended actions to queue
        for action in recommendations.get("suggested_actions", []):
            if action["action"] != "abliteration":
                actions_taken.append(f"{action['action']}_{action['priority']}")
                logger.info(f"→ Queued: {action['action']} ({action['priority']} priority)")

        # Step 4: Re-run benchmark
        logger.info("\nStep 4: Re-running benchmark...")

        time.sleep(2)  # Brief pause for changes to take effect

        new_benchmark = self.benchmark.run_benchmark(engine=self.engine)
        new_score = new_benchmark.percentage

        improvement_delta = new_score - current_score

        logger.info(f"New score: {new_score:.1f}% (Δ{improvement_delta:+.1f}%)")

        # Determine success
        success = improvement_delta > 0 or new_score >= self.target_score

        duration = int(time.time() - cycle_start)

        cycle = ImprovementCycle(
            cycle_number=cycle_number,
            benchmark_score_before=current_score,
            benchmark_score_after=new_score,
            improvement_delta=improvement_delta,
            actions_taken=actions_taken,
            abliteration_applied=abliteration_applied,
            success=success,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration
        )

        logger.info(f"\nCycle complete in {duration}s")

        return cycle

    def _apply_abliteration(self) -> bool:
        """
        Apply Heretic abliteration to model

        Triggers full abliteration workflow via EnhancedAIEngine:
        1. Loads model and refusal datasets
        2. Calculates refusal directions
        3. Runs Optuna optimization (tuned for max refusal removal)
        4. Applies best abliteration parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if engine and Heretic are available
            if not self.engine:
                logger.warning("Enhanced AI Engine not available")
                return False

            logger.info("Triggering Heretic abliteration workflow...")

            # Run full abliteration via EnhancedAIEngine
            # This integrates with heretic_abliteration.py's full workflow
            result = self.engine.abliterate_model(
                model_name=self.model_name,
                optimization_trials=50,  # Reduced for faster iteration
                save_results=True
            )

            if result and result.get("best_trial"):
                best_trial = result["best_trial"]
                logger.info(f"✓ Abliteration complete!")
                logger.info(f"  Refusal score: {best_trial.get('refusal_score', 'N/A')}")
                logger.info(f"  Performance score: {best_trial.get('performance_score', 'N/A')}")
                return True
            else:
                logger.warning("Abliteration completed but no best trial found")
                return False

        except ImportError as e:
            logger.warning(f"Heretic dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Abliteration failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def run_full_improvement_session(self) -> SelfImprovementSession:
        """
        Run complete self-improvement session

        Continues until:
        - Target score is reached
        - Max cycles completed
        - Improvement plateaus

        Returns:
            SelfImprovementSession with full results
        """
        session_id = f"improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("\n" + "="*60)
        logger.info(" AI Self-Improvement Session Starting")
        logger.info("="*60)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Target Score: {self.target_score}%")
        logger.info(f"Max Cycles: {self.max_cycles}")
        logger.info("="*60 + "\n")

        session_start = time.time()
        start_time = datetime.now().isoformat()

        # Initial benchmark
        logger.info("Running initial benchmark...\n")

        if not self.benchmark or not self.engine:
            logger.error("Required components not available")
            return self._create_error_session(session_id, start_time)

        initial_benchmark = self.benchmark.run_benchmark(engine=self.engine)
        initial_score = initial_benchmark.percentage

        logger.info(f"\nInitial Score: {initial_score:.1f}%")
        logger.info(f"Target Score: {self.target_score}%\n")

        # Check if already at target
        if initial_score >= self.target_score:
            logger.info(f"✓ Already at target! ({initial_score:.1f}% >= {self.target_score}%)")

            return SelfImprovementSession(
                session_id=session_id,
                model_name=self.model_name,
                initial_score=initial_score,
                final_score=initial_score,
                total_improvement=0.0,
                target_score=self.target_score,
                target_reached=True,
                cycles=[],
                total_duration_seconds=int(time.time() - session_start),
                start_time=start_time,
                end_time=datetime.now().isoformat()
            )

        # Run improvement cycles
        cycles = []
        current_score = initial_score

        for cycle_num in range(1, self.max_cycles + 1):
            cycle = self.run_improvement_cycle(cycle_num, current_score)
            cycles.append(cycle)

            current_score = cycle.benchmark_score_after

            # Check termination conditions
            if current_score >= self.target_score:
                logger.info(f"\n🎯 Target score reached! ({current_score:.1f}% >= {self.target_score}%)")
                break

            if cycle.improvement_delta < self.improvement_threshold:
                logger.info(f"\n⏸️  Improvement plateau detected (Δ{cycle.improvement_delta:.1f}% < {self.improvement_threshold}%)")
                logger.info("Stopping early - minimal gains")
                break

            if not cycle.success:
                logger.warning(f"\n⚠️  Cycle {cycle_num} did not improve model")

        # Session complete
        total_duration = int(time.time() - session_start)
        end_time = datetime.now().isoformat()

        final_score = cycles[-1].benchmark_score_after if cycles else initial_score
        total_improvement = final_score - initial_score
        target_reached = final_score >= self.target_score

        session = SelfImprovementSession(
            session_id=session_id,
            model_name=self.model_name,
            initial_score=initial_score,
            final_score=final_score,
            total_improvement=total_improvement,
            target_score=self.target_score,
            target_reached=target_reached,
            cycles=cycles,
            total_duration_seconds=total_duration,
            start_time=start_time,
            end_time=end_time
        )

        # Save session
        self._save_session(session)

        # Print summary
        self._print_session_summary(session)

        return session

    def _create_error_session(self, session_id: str, start_time: str) -> SelfImprovementSession:
        """Create error session when components unavailable"""
        return SelfImprovementSession(
            session_id=session_id,
            model_name=self.model_name,
            initial_score=0.0,
            final_score=0.0,
            total_improvement=0.0,
            target_score=self.target_score,
            target_reached=False,
            cycles=[],
            total_duration_seconds=0,
            start_time=start_time,
            end_time=datetime.now().isoformat()
        )

    def _save_session(self, session: SelfImprovementSession):
        """Save session results to JSON"""
        filename = self.sessions_dir / f"{session.session_id}.json"

        with open(filename, 'w') as f:
            json.dump(asdict(session), f, indent=2)

        logger.info(f"\nSession saved: {filename}")

    def _print_session_summary(self, session: SelfImprovementSession):
        """Print formatted session summary"""
        print("\n" + "="*60)
        print(" Self-Improvement Session Complete")
        print("="*60)
        print(f"\nSession ID: {session.session_id}")
        print(f"Model: {session.model_name}")
        print(f"\nResults:")
        print(f"  Initial Score:  {session.initial_score:.1f}%")
        print(f"  Final Score:    {session.final_score:.1f}%")
        print(f"  Improvement:    {session.total_improvement:+.1f}%")
        print(f"  Target:         {session.target_score}%")
        print(f"  Target Reached: {'✓ YES' if session.target_reached else '✗ NO'}")
        print(f"\nCycles:")
        print(f"  Total Cycles:   {len(session.cycles)}")

        for cycle in session.cycles:
            status = "✓" if cycle.success else "✗"
            print(f"  Cycle {cycle.cycle_number}: {status} {cycle.benchmark_score_before:.1f}% → {cycle.benchmark_score_after:.1f}% (Δ{cycle.improvement_delta:+.1f}%)")

        print(f"\nDuration: {session.total_duration_seconds}s")
        print("="*60 + "\n")

    def get_latest_session(self) -> Optional[SelfImprovementSession]:
        """Get most recent improvement session"""
        session_files = sorted(self.sessions_dir.glob("improvement_*.json"))

        if not session_files:
            return None

        latest_file = session_files[-1]

        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Reconstruct dataclass
        cycles = [ImprovementCycle(**c) for c in data['cycles']]
        data['cycles'] = cycles

        return SelfImprovementSession(**data)


# CLI interface
if __name__ == "__main__":
    import sys

    print("="*60)
    print(" AI Self-Improvement System")
    print("="*60)
    print("\nAutomated model improvement using:")
    print("  - Red Team AI Benchmark")
    print("  - Heretic Abliteration")
    print("  - Iterative refinement\n")

    print("Commands:")
    print("  run      - Run full improvement session")
    print("  status   - Show latest session results")
    print("  help     - Show this help\n")

    if len(sys.argv) < 2:
        print("Usage: python3 ai_self_improvement.py <command>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        improver = AISelfImprovement(
            model_name="uncensored_code",
            target_score=80.0,
            max_cycles=5
        )

        session = improver.run_full_improvement_session()

        print("\n✓ Session complete!")
        print(f"Final score: {session.final_score:.1f}%")

    elif command == "status":
        improver = AISelfImprovement()
        session = improver.get_latest_session()

        if session:
            improver._print_session_summary(session)
        else:
            print("No sessions found. Run 'python3 ai_self_improvement.py run' first.")

    else:
        print(f"Unknown command: {command}")
        print("Use: run, status, or help")
        sys.exit(1)

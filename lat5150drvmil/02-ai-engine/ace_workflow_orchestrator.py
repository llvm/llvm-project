#!/usr/bin/env python3
"""
ACE-FCA Phase-Based Workflow Orchestrator

Implements structured Research → Plan → Implement → Verify workflow
with human review checkpoints at compaction boundaries.

Key features:
- Phase isolation with context compaction between phases
- Specialized subagents for each phase
- Human-in-the-loop review at critical boundaries
- Automatic context management (40-60% utilization)
- Comprehensive logging and progress tracking

Based on ACE-FCA methodology:
https://github.com/humanlayer/advanced-context-engineering-for-coding-agents
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ace_context_engine import (
    ACEContextEngine,
    PhaseType,
    PhaseOutput,
    ContextBlock
)
from ace_config import ACEConfiguration, load_configuration
from ace_exceptions import (
    ACEError,
    PhaseExecutionError,
    ConfigurationError,
    ValidationError,
    format_error_for_logging
)
from ace_interfaces import ReviewInterface, InteractiveReview
import logging

logger = logging.getLogger(__name__)


class SubagentRole(Enum):
    """Specialized subagent roles for context isolation"""
    RESEARCHER = "researcher"         # Search, file analysis, architecture understanding
    PLANNER = "planner"              # Create implementation plans
    IMPLEMENTER = "implementer"      # Execute code changes
    VERIFIER = "verifier"            # Test and validate
    SUMMARIZER = "summarizer"        # Compress and summarize outputs


@dataclass
class WorkflowTask:
    """Represents a coding task to be executed through workflow"""
    description: str
    task_type: str = "feature"  # 'feature', 'bugfix', 'refactor', 'analysis'
    priority: int = 5
    estimated_complexity: str = "medium"  # 'simple', 'medium', 'complex'
    constraints: List[str] = field(default_factory=list)
    context_files: List[str] = field(default_factory=list)


@dataclass
class ReviewCheckpoint:
    """Human review checkpoint at compaction boundary"""
    phase: PhaseType
    content: str
    requires_approval: bool = True
    approved: bool = False
    feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase.value,
            "content": self.content,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat()
        }


class ACEWorkflowOrchestrator:
    """
    Phase-based workflow orchestrator implementing ACE-FCA patterns

    Workflow: Research → Plan → Implement → Verify
    Each phase has:
    - Dedicated subagent for context isolation
    - Automatic compaction at phase boundaries
    - Optional human review checkpoints
    """

    def __init__(self,
                 ai_engine,  # DSMILAIEngine instance
                 config: Optional[ACEConfiguration] = None,
                 review_interface: Optional[ReviewInterface] = None,
                 max_tokens: Optional[int] = None,
                 enable_human_review: bool = True):
        """
        Initialize workflow orchestrator

        Args:
            ai_engine: AI engine for model inference
            config: ACE configuration (uses default if None)
            review_interface: Review interface for human review (creates default if None)
            max_tokens: Maximum context window size (overrides config if provided)
            enable_human_review: Enable human review checkpoints
        """
        self.ai_engine = ai_engine

        # Load configuration
        self.config = config or load_configuration()

        # Override max_tokens if provided
        if max_tokens is not None:
            self.config.max_context_tokens = max_tokens

        self.ace = ACEContextEngine(max_tokens=self.config.max_context_tokens)

        # Setup review interface
        self.review_interface = review_interface or InteractiveReview()
        self.enable_human_review = enable_human_review
        if not enable_human_review:
            self.review_interface.set_auto_approve(True)

        self.current_task: Optional[WorkflowTask] = None
        self.review_checkpoints: List[ReviewCheckpoint] = []
        self.execution_log: List[Dict] = []

        # Phase-specific prompts from configuration
        self.phase_prompts = {
            PhaseType.RESEARCH: self.config.get_phase_config(PhaseType.RESEARCH).prompt_template,
            PhaseType.PLAN: self.config.get_phase_config(PhaseType.PLAN).prompt_template,
            PhaseType.IMPLEMENT: self.config.get_phase_config(PhaseType.IMPLEMENT).prompt_template,
            PhaseType.VERIFY: self.config.get_phase_config(PhaseType.VERIFY).prompt_template,
        }

        logger.info(f"ACE Workflow Orchestrator initialized with config: {self.config.max_context_tokens} max tokens")

    def execute_task(self,
                    task: WorkflowTask,
                    model_preference: str = "quality_code") -> Dict:
        """
        Execute a complete workflow for a coding task

        Args:
            task: WorkflowTask to execute
            model_preference: Model to use ('fast', 'code', 'quality_code', etc.)

        Returns:
            Dict with execution results and metadata
        """
        self.current_task = task
        self.review_checkpoints = []
        self.execution_log = []

        print(f"\n{'='*60}")
        print(f"ACE-FCA Workflow: {task.description}")
        print(f"{'='*60}\n")

        try:
            # Phase 1: RESEARCH
            research_result = self._execute_phase(
                PhaseType.RESEARCH,
                model_preference=model_preference
            )

            # Review checkpoint after research
            if self.enable_human_review:
                review_approved = self._create_review_checkpoint(
                    PhaseType.RESEARCH,
                    research_result["output"]
                )
                if not review_approved:
                    return {
                        "success": False,
                        "phase_stopped": "research",
                        "reason": "Review not approved"
                    }

            # Phase 2: PLAN
            plan_result = self._execute_phase(
                PhaseType.PLAN,
                model_preference=model_preference
            )

            # Review checkpoint after planning
            if self.enable_human_review:
                review_approved = self._create_review_checkpoint(
                    PhaseType.PLAN,
                    plan_result["output"]
                )
                if not review_approved:
                    return {
                        "success": False,
                        "phase_stopped": "plan",
                        "reason": "Review not approved"
                    }

            # Phase 3: IMPLEMENT
            implement_result = self._execute_phase(
                PhaseType.IMPLEMENT,
                model_preference=model_preference
            )

            # Phase 4: VERIFY
            verify_result = self._execute_phase(
                PhaseType.VERIFY,
                model_preference=model_preference
            )

            # Final review checkpoint
            if self.enable_human_review:
                review_approved = self._create_review_checkpoint(
                    PhaseType.VERIFY,
                    verify_result["output"]
                )

            return {
                "success": True,
                "task": task.description,
                "phases_completed": [
                    "research", "plan", "implement", "verify"
                ],
                "research_output": research_result["output"],
                "plan_output": plan_result["output"],
                "implementation_notes": implement_result["output"],
                "verification_results": verify_result["output"],
                "context_stats": self.ace.get_stats(),
                "review_checkpoints": [r.to_dict() for r in self.review_checkpoints],
                "execution_log": self.execution_log
            }

        except ACEError as e:
            logger.error(format_error_for_logging(e))
            return {
                "success": False,
                "error": str(e),
                "error_details": e.to_dict(),
                "phase": self.ace.current_phase.value if self.ace.current_phase else None,
                "execution_log": self.execution_log
            }
        except Exception as e:
            logger.error(f"Unexpected error in workflow: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "phase": self.ace.current_phase.value if self.ace.current_phase else None,
                "execution_log": self.execution_log
            }

    def _execute_phase(self,
                      phase: PhaseType,
                      model_preference: str = "quality_code") -> Dict:
        """
        Execute a single workflow phase

        Returns:
            Dict with phase output and metadata
        """
        print(f"\n🔹 Phase: {phase.value.upper()}")
        print(f"   Context: {self.ace.get_stats()['utilization_percent']}")

        # Start phase (triggers compaction if needed)
        phase_start = self.ace.start_phase(phase)
        if phase_start["compaction_performed"]:
            print(f"   ✓ Context compacted: {phase_start['compaction_stats']['tokens_freed']} tokens freed")

        # Build phase-specific prompt
        phase_prompt = self.phase_prompts[phase]

        # Add task context
        task_context = f"""
Task: {self.current_task.description}
Type: {self.current_task.task_type}
Complexity: {self.current_task.estimated_complexity}
"""
        if self.current_task.constraints:
            task_context += f"\nConstraints:\n" + "\n".join(f"- {c}" for c in self.current_task.constraints)

        # Add previous phase outputs for context
        previous_context = ""
        if phase == PhaseType.PLAN:
            research_output = self.ace.get_phase_output(PhaseType.RESEARCH)
            if research_output:
                previous_context = f"\n## Research Findings:\n{research_output.content}"

        elif phase == PhaseType.IMPLEMENT:
            plan_output = self.ace.get_phase_output(PhaseType.PLAN)
            if plan_output:
                previous_context = f"\n## Implementation Plan:\n{plan_output.content}"

        elif phase == PhaseType.VERIFY:
            implement_output = self.ace.get_phase_output(PhaseType.IMPLEMENT)
            if implement_output:
                previous_context = f"\n## Implementation Notes:\n{implement_output.content}"

        # Build full prompt using ACE context engine
        full_prompt = self.ace.build_prompt(
            user_query=f"{phase_prompt}\n{task_context}\n{previous_context}"
        )

        # Execute with AI engine
        print(f"   Running {phase.value} agent...")
        response = self.ai_engine.generate(
            full_prompt,
            model=model_preference,
            stream=False
        )

        output = response.get("text", "")

        # Complete phase and store output
        phase_output = self.ace.complete_phase(
            output=output,
            metadata={
                "model": model_preference,
                "tokens": len(output.split()),
                "timestamp": datetime.now().isoformat()
            }
        )

        # Log execution
        self.execution_log.append({
            "phase": phase.value,
            "timestamp": datetime.now().isoformat(),
            "tokens": phase_output.token_count,
            "output_preview": output[:200] + "..." if len(output) > 200 else output
        })

        print(f"   ✓ {phase.value.capitalize()} complete ({phase_output.token_count} tokens)")

        return {
            "phase": phase.value,
            "output": output,
            "metadata": phase_output.metadata
        }

    def _create_review_checkpoint(self,
                                 phase: PhaseType,
                                 content: str) -> bool:
        """
        Create human review checkpoint at compaction boundary

        Args:
            phase: Phase being reviewed
            content: Content to review

        Returns:
            True if approved, False if rejected
        """
        from ace_interfaces import ReviewRequest, ReviewStatus

        checkpoint = ReviewCheckpoint(
            phase=phase,
            content=content,
            requires_approval=True
        )

        print(f"\n📋 REVIEW CHECKPOINT: {phase.value.upper()} Phase")
        print(f"   Context: {self.ace.get_stats()['utilization_percent']}")

        # Use review interface
        review_request = ReviewRequest(
            title=f"{phase.value.upper()} Phase Review",
            content=content,
            context={
                "phase": phase.value,
                "utilization": self.ace.get_stats()['utilization_percent'],
                "task": self.current_task.description if self.current_task else None
            }
        )

        review_result = self.review_interface.request_review(review_request)

        checkpoint.approved = review_result.approved
        checkpoint.feedback = review_result.feedback
        self.review_checkpoints.append(checkpoint)

        if review_result.approved:
            print("   ✓ Approved - continuing to next phase")
        else:
            print("   ✗ Rejected - stopping workflow")

        return review_result.approved

    def get_workflow_summary(self) -> Dict:
        """Get comprehensive workflow execution summary"""
        return {
            "task": self.current_task.description if self.current_task else None,
            "context_stats": self.ace.get_stats(),
            "phases_completed": [p.phase.value for p in self.ace.phase_outputs],
            "review_checkpoints": [r.to_dict() for r in self.review_checkpoints],
            "execution_log": self.execution_log
        }


# Example usage
if __name__ == "__main__":
    from tests.ace_test_utils import MockAIEngine

    print("ACE-FCA Workflow Orchestrator")
    print("=" * 60)

    # Test workflow with mock AI engine
    ai_engine = MockAIEngine()
    orchestrator = ACEWorkflowOrchestrator(
        ai_engine=ai_engine,
        enable_human_review=False  # Disable for automated testing
    )

    task = WorkflowTask(
        description="Add rate limiting to API endpoints",
        task_type="feature",
        estimated_complexity="medium",
        constraints=["Must not break existing endpoints", "Must be configurable"]
    )

    result = orchestrator.execute_task(task, model_preference="code")

    print("\n" + "="*60)
    print("WORKFLOW RESULTS:")
    print("="*60)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Phases: {', '.join(result['phases_completed'])}")
        print(f"\nContext Stats:")
        for key, value in result['context_stats'].items():
            print(f"  {key}: {value}")

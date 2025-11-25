#!/usr/bin/env python3
"""
Integrated Local Claude Code System
Complete agentic coding system with self-coding capabilities and incremental learning

This system can code itself and learn from codebases as it processes them.
"""

import sys
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import core components
from dsmil_ai_engine import DSMILAIEngine
from file_operations import FileOps
from edit_operations import EditOps
from tool_operations import ToolOps
from advanced_planner import AdvancedPlanner, TaskComplexity
from execution_engine import ExecutionEngine, ExecutionStatus
from context_manager import ContextManager
from codebase_learner import CodebaseLearner
from pattern_database import PatternDatabase, PatternCategory, PatternQuality

# Import Serena semantic code engine
sys.path.insert(0, str(Path(__file__).parent.parent / "01-source" / "serena_integration"))
try:
    from semantic_code_engine import SemanticCodeEngine
    SERENA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Serena semantic engine not available: {e}")
    SERENA_AVAILABLE = False

# Import existing integrated systems
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations" / "rag_system"))

try:
    from jina_v3_retriever import JinaV3Retriever
    from int8_auto_coding import INT8AutoCoding, create_int8_coding_system
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG/Auto-coding not available: {e}")
    RAG_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedLocalClaude:
    """
    Complete integrated agentic coding system

    Features:
    - Plan and execute multi-step coding tasks
    - Learn from codebases incrementally
    - Use RAG for context retrieval
    - INT8-optimized code generation
    - Self-coding capabilities
    - Pattern database for best practices
    - Session persistence
    - Error recovery and retries
    """

    def __init__(
        self,
        workspace_root: str = ".",
        enable_rag: bool = True,
        enable_int8_coding: bool = True,
        enable_learning: bool = True,
        int8_preset: str = "int8_balanced"
    ):
        """
        Initialize integrated system

        Args:
            workspace_root: Project root directory
            enable_rag: Enable RAG for context retrieval
            enable_int8_coding: Enable INT8-optimized code generation
            enable_learning: Enable incremental learning
            int8_preset: INT8 configuration preset
        """
        self.workspace_root = Path(workspace_root).resolve()

        logger.info("="*80)
        logger.info("INTEGRATED LOCAL CLAUDE CODE")
        logger.info("="*80)
        logger.info(f"Workspace: {self.workspace_root}")

        # Core components
        logger.info("Initializing core components...")
        self.ai = DSMILAIEngine()
        self.files = FileOps(workspace_root=str(self.workspace_root))
        self.edits = EditOps(workspace_root=str(self.workspace_root), create_backups=True)
        self.tools = ToolOps(workspace_root=str(self.workspace_root))
        self.context = ContextManager(workspace_root=str(self.workspace_root))

        # Pattern database
        logger.info("Initializing pattern database...")
        self.pattern_db = PatternDatabase(rag_system=None)  # Will add RAG later

        # Codebase learner
        if enable_learning:
            logger.info("Initializing codebase learner...")
            self.learner = CodebaseLearner(
                workspace_root=str(self.workspace_root),
                pattern_db=self.pattern_db
            )
        else:
            self.learner = None

        # RAG system
        if enable_rag and RAG_AVAILABLE:
            logger.info("Initializing RAG system (Jina v3)...")
            try:
                self.rag = JinaV3Retriever()
                self.pattern_db.rag = self.rag
                if self.learner:
                    self.learner.rag = self.rag
                logger.info("✓ RAG system ready")
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
                self.rag = None
        else:
            self.rag = None

        # Serena semantic code engine (symbol-level code understanding)
        if SERENA_AVAILABLE:
            logger.info("Initializing Serena semantic code engine...")
            self.serena = SemanticCodeEngine(workspace_path=str(self.workspace_root))
            # Initialize asynchronously when first used
            logger.info("✓ Serena semantic engine ready (LSP-based)")
        else:
            self.serena = None
            logger.warning("⚠️  Serena not available - using basic file operations only")

        # Planner
        logger.info("Initializing advanced planner...")
        self.planner = AdvancedPlanner(
            ai_engine=self.ai,
            rag_system=self.rag,
            pattern_db=self.pattern_db
        )

        # Execution engine
        logger.info("Initializing execution engine...")
        self.executor = ExecutionEngine(
            ai_engine=self.ai,
            file_ops=self.files,
            edit_ops=self.edits,
            tool_ops=self.tools,
            context_manager=self.context,
            codebase_learner=self.learner,
            pattern_db=self.pattern_db,
            semantic_engine=self.serena if SERENA_AVAILABLE else None,
            max_retries=2
        )

        # INT8 auto-coding system
        if enable_int8_coding and RAG_AVAILABLE:
            logger.info(f"Initializing INT8 auto-coding ({int8_preset})...")
            try:
                self.int8_coder = create_int8_coding_system(preset=int8_preset)
                logger.info("✓ INT8 auto-coding ready")
            except Exception as e:
                logger.warning(f"Could not initialize INT8 coding: {e}")
                self.int8_coder = None
        else:
            self.int8_coder = None

        # Initialize knowledge
        self._initialize_knowledge()

        logger.info("="*80)
        logger.info("SYSTEM READY")
        logger.info("="*80)
        logger.info("Capabilities:")
        logger.info("  ✓ Multi-step task planning and execution")
        logger.info("  ✓ File reading, editing, and writing")
        logger.info("  ✓ Command execution and testing")
        logger.info("  ✓ Context tracking and session persistence")
        logger.info(f"  {'✓' if self.learner else '✗'} Incremental codebase learning")
        logger.info(f"  {'✓' if self.rag else '✗'} RAG-enhanced context retrieval")
        logger.info(f"  {'✓' if self.int8_coder else '✗'} INT8-optimized code generation")
        logger.info(f"  {'✓' if self.pattern_db else '✗'} Pattern database and best practices")
        logger.info("  ✓ Self-coding capabilities")
        logger.info("  ✓ Error recovery and retries")
        logger.info("="*80 + "\n")

    def _initialize_knowledge(self):
        """Initialize knowledge from existing codebase"""
        logger.info("Initializing knowledge...")

        if self.learner:
            # Learn from current Python files
            python_files = list(self.workspace_root.rglob("*.py"))

            if python_files:
                logger.info(f"Found {len(python_files)} Python files")
                logger.info("Learning from existing codebase (sampling)...")

                # Sample up to 20 files for initial learning
                sample_size = min(20, len(python_files))
                import random
                sampled = random.sample(python_files, sample_size)

                for filepath in sampled:
                    try:
                        result = self.learner.learn_from_file(str(filepath))
                        if 'error' not in result:
                            logger.debug(f"  Learned from {filepath.name}: {result.get('functions', 0)} functions, {result.get('classes', 0)} classes")
                    except Exception as e:
                        logger.debug(f"  Could not learn from {filepath.name}: {e}")

                stats = self.learner.get_learning_stats()
                logger.info(f"✓ Learned {stats['patterns_learned']} patterns from codebase")

    def execute_task(
        self,
        task: str,
        dry_run: bool = False,
        interactive: bool = False,
        use_int8_coding: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a coding task

        Args:
            task: Task description
            dry_run: If True, simulate execution without making changes
            interactive: If True, prompt before each step
            use_int8_coding: Use INT8-optimized code generation if available

        Returns:
            Execution results
        """
        logger.info("\n" + "="*80)
        logger.info(f"TASK: {task}")
        logger.info("="*80 + "\n")

        # Build context
        context = {
            "workspace": str(self.workspace_root),
            "recent_files": list(self.context.files_accessed.keys())[-10:],
            "conversation_history": len(self.context.conversation_history)
        }

        # Plan the task
        logger.info("📋 Planning task...")
        plan = self.planner.plan_task(task, context=context)

        logger.info(f"\n📝 Plan created:")
        logger.info(f"   Complexity: {plan.complexity.value}")
        logger.info(f"   Steps: {len(plan.steps)}")
        logger.info(f"   Estimated time: {plan.estimated_time}s")
        logger.info(f"   Model: {plan.model_used}")

        if plan.files_involved:
            logger.info(f"   Files: {', '.join(plan.files_involved[:5])}")

        if plan.risks:
            logger.info(f"\n⚠️  Risks:")
            for risk in plan.risks:
                logger.info(f"   - {risk}")

        logger.info(f"\n📊 Steps:")
        for step in plan.steps:
            logger.info(f"   {step.step_num}. [{step.step_type.value}] {step.description}")

        # Execute the plan
        logger.info(f"\n⚡ Executing plan...")
        result = self.executor.execute_plan(plan, dry_run=dry_run, interactive=interactive)

        # Log results
        logger.info(f"\n" + "="*80)
        logger.info(f"EXECUTION COMPLETE: {result.status.value.upper()}")
        logger.info("="*80)
        logger.info(f"Duration: {result.total_duration:.1f}s")
        logger.info(f"Steps succeeded: {result.steps_succeeded}/{len(plan.steps)}")
        logger.info(f"Steps failed: {result.steps_failed}/{len(plan.steps)}")

        if result.learned_patterns > 0:
            logger.info(f"Patterns learned: {result.learned_patterns}")

        # Record in context
        self.context.add_conversation_turn(
            user_message=task,
            assistant_response=f"Executed {len(plan.steps)} steps",
            task_type=plan.complexity.value,
            files_involved=plan.files_involved,
            success=result.status == ExecutionStatus.SUCCESS,
            model_used=plan.model_used
        )

        # Record in planner
        self.planner.record_execution(plan, [r.__dict__ for r in result.step_results], result.status == ExecutionStatus.SUCCESS)

        # Save knowledge
        if self.learner:
            self.learner.save_knowledge()

        # Save session
        self.context.save_session()

        return {
            "task": task,
            "status": result.status.value,
            "steps": len(plan.steps),
            "succeeded": result.steps_succeeded,
            "failed": result.steps_failed,
            "duration": result.total_duration,
            "learned_patterns": result.learned_patterns,
            "dry_run": dry_run
        }

    def code_itself(
        self,
        improvement_description: str,
        target_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Self-coding: System codes and improves itself

        Args:
            improvement_description: What to improve
            target_file: Specific file to modify (optional)

        Returns:
            Results of self-modification
        """
        logger.info("\n" + "="*80)
        logger.info("SELF-CODING MODE")
        logger.info("="*80)
        logger.info(f"Improvement: {improvement_description}\n")

        # Determine target
        if not target_file:
            # Find the most relevant file
            task = f"Implement: {improvement_description}"

            if self.rag:
                results = self.rag.search(task, top_k=5)
                target_file = results[0].metadata.get('filepath') if results else None

            if not target_file:
                logger.warning("No target file specified or found")
                return {"error": "No target file"}

        logger.info(f"Target: {target_file}")

        # Create self-modification task
        task = f"Modify {target_file} to: {improvement_description}"

        # Execute with extra caution
        logger.info("\n⚠️  SELF-MODIFICATION REQUIRES REVIEW")
        logger.info("The system will modify its own code. Review changes carefully.\n")

        result = self.execute_task(task, dry_run=False, interactive=True)

        logger.info("\n✓ Self-modification complete")
        logger.info("Review changes before committing!")

        return result

    def learn_from_codebase(
        self,
        path: Optional[str] = None,
        file_pattern: str = "**/*.py",
        max_files: int = 100
    ) -> Dict[str, Any]:
        """
        Learn from entire codebase

        Args:
            path: Path to codebase (default: workspace_root)
            file_pattern: File pattern to match
            max_files: Maximum files to process

        Returns:
            Learning statistics
        """
        if not self.learner:
            return {"error": "Learning not enabled"}

        logger.info("\n" + "="*80)
        logger.info("CODEBASE LEARNING")
        logger.info("="*80)

        search_path = Path(path) if path else self.workspace_root
        logger.info(f"Path: {search_path}")
        logger.info(f"Pattern: {file_pattern}\n")

        # Find files
        files = list(search_path.glob(file_pattern))[:max_files]

        logger.info(f"Found {len(files)} files to analyze")

        # Learn from each file
        total_functions = 0
        total_classes = 0
        total_patterns = 0

        for i, filepath in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] Learning from {filepath.name}...")

            try:
                result = self.learner.learn_from_file(str(filepath))

                if 'error' not in result:
                    total_functions += result.get('functions', 0)
                    total_classes += result.get('classes', 0)
                    total_patterns += len(result.get('patterns', []))

            except Exception as e:
                logger.error(f"  Error: {e}")

        # Save knowledge
        self.learner.save_knowledge()

        logger.info("\n" + "="*80)
        logger.info("LEARNING COMPLETE")
        logger.info("="*80)
        logger.info(f"Files analyzed: {len(files)}")
        logger.info(f"Functions learned: {total_functions}")
        logger.info(f"Classes learned: {total_classes}")
        logger.info(f"Total patterns: {total_patterns}")

        style_guide = self.learner.get_style_guide()
        logger.info(f"\nCoding style detected:")
        logger.info(f"  Indentation: {style_guide['indentation']}")
        logger.info(f"  Quotes: {style_guide['quotes']}")
        logger.info(f"  Naming: {style_guide['naming_conventions']}")

        return {
            "files_analyzed": len(files),
            "functions_learned": total_functions,
            "classes_learned": total_classes,
            "patterns_learned": total_patterns,
            "style_guide": style_guide
        }

    def import_best_practices(self, json_path: str) -> int:
        """
        Import best practices from JSON file

        Args:
            json_path: Path to JSON file with patterns and practices

        Returns:
            Number of items imported
        """
        logger.info(f"Importing best practices from {json_path}...")

        count = self.pattern_db.import_patterns_from_json(json_path)

        logger.info(f"✓ Imported {count} items")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "session": self.context.get_session_stats(),
            "execution": self.planner.get_execution_stats(),
            "database": self.pattern_db.get_database_stats()
        }

        if self.learner:
            stats["learning"] = self.learner.get_learning_stats()

        return stats

    def save_state(self):
        """Save all system state"""
        logger.info("Saving system state...")

        self.context.save_session()

        if self.learner:
            self.learner.save_knowledge()

        logger.info("✓ State saved")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrated Local Claude Code - Agentic Coding System"
    )
    parser.add_argument("task", nargs="*", help="Task to execute")
    parser.add_argument("--workspace", default=".", help="Workspace root directory")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--learn", action="store_true", help="Learn from codebase")
    parser.add_argument("--self-code", metavar="IMPROVEMENT", help="Self-coding mode")
    parser.add_argument("--import-practices", metavar="JSON", help="Import best practices from JSON")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 coding")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning")

    args = parser.parse_args()

    # Initialize system
    system = IntegratedLocalClaude(
        workspace_root=args.workspace,
        enable_rag=not args.no_rag,
        enable_int8_coding=not args.no_int8,
        enable_learning=not args.no_learning
    )

    # Execute command
    if args.stats:
        # Show statistics
        stats = system.get_stats()
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        print(json.dumps(stats, indent=2))

    elif args.learn:
        # Learn from codebase
        result = system.learn_from_codebase()
        print(f"\n✓ Learning complete: {result['patterns_learned']} patterns")

    elif args.self_code:
        # Self-coding mode
        result = system.code_itself(args.self_code)
        print(f"\n✓ Self-modification: {result.get('status', 'unknown')}")

    elif args.import_practices:
        # Import best practices
        count = system.import_best_practices(args.import_practices)
        print(f"\n✓ Imported {count} items")

    elif args.task:
        # Execute task
        task = " ".join(args.task)
        result = system.execute_task(
            task,
            dry_run=args.dry_run,
            interactive=args.interactive
        )

        print(f"\n✓ Task {result['status']}: {result['succeeded']}/{result['steps']} steps succeeded")

    else:
        # Show help
        parser.print_help()
        print("\nExamples:")
        print("  # Execute a task")
        print("  python integrated_local_claude.py 'Add logging to server.py'")
        print("\n  # Dry run")
        print("  python integrated_local_claude.py --dry-run 'Refactor database.py'")
        print("\n  # Interactive mode")
        print("  python integrated_local_claude.py -i 'Fix authentication bug'")
        print("\n  # Learn from codebase")
        print("  python integrated_local_claude.py --learn")
        print("\n  # Self-coding")
        print("  python integrated_local_claude.py --self-code 'Add better error handling'")
        print("\n  # Show statistics")
        print("  python integrated_local_claude.py --stats")


if __name__ == "__main__":
    main()

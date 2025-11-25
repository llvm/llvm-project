#!/usr/bin/env python3
"""
Natural Language Interface for Integrated Local Claude Code
Provides conversational interface to self-coding system with streaming responses
"""

import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Generator, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Import integrated system
sys.path.insert(0, str(Path(__file__).parent))
from integrated_local_claude import IntegratedLocalClaude
from advanced_planner import ExecutionPlan, TaskComplexity
from execution_engine import ExecutionStatus

# Import unified orchestrator for automatic search routing
try:
    from unified_orchestrator import UnifiedAIOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Import NotebookLM subagent for document research
try:
    from notebooklm_subagent import NotebookLMSubagent
    NOTEBOOKLM_AVAILABLE = True
except ImportError:
    NOTEBOOKLM_AVAILABLE = False

# Import workflow automation hub
try:
    from workflow_automation_hub import WorkflowAutomationHub, WorkflowType, parse_workflow_request
    WORKFLOW_HUB_AVAILABLE = True
except ImportError:
    WORKFLOW_HUB_AVAILABLE = False

# Import context optimizer for intelligent context window management
try:
    from context_optimizer_integration import ContextOptimizerIntegration
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError:
    CONTEXT_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    STATUS = "status"
    ERROR = "error"


class DisplayEventType(Enum):
    """Types of display events for streaming"""
    PLANNING = "planning"
    EXECUTING = "executing"
    READING = "reading"
    EDITING = "editing"
    WRITING = "writing"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    TESTING = "testing"
    LEARNING = "learning"
    COMPLETE = "complete"
    ERROR = "error"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    PROGRESS = "progress"


@dataclass
class DisplayEvent:
    """Event for display streaming"""
    event_type: DisplayEventType
    message: str
    data: Dict[str, Any] = None
    timestamp: float = None
    progress: Optional[float] = None  # 0.0 to 1.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.data is None:
            self.data = {}

    def to_json(self) -> str:
        """Convert to JSON for streaming"""
        return json.dumps({
            "type": self.event_type.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "progress": self.progress
        })


@dataclass
class ConversationMessage:
    """Single message in conversation"""
    role: MessageType
    content: str
    timestamp: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class NaturalLanguageInterface:
    """
    Natural language interface for self-coding system

    Features:
    - Conversational interaction
    - Intent recognition
    - Streaming responses
    - Visual progress feedback
    - Multi-turn conversations
    - Context retention
    """

    def __init__(
        self,
        workspace_root: str = ".",
        enable_rag: bool = True,
        enable_int8: bool = True,
        enable_learning: bool = True
    ):
        """
        Initialize natural language interface

        Args:
            workspace_root: Project root directory
            enable_rag: Enable RAG for context retrieval
            enable_int8: Enable INT8-optimized code generation
            enable_learning: Enable incremental learning
        """
        self.system = IntegratedLocalClaude(
            workspace_root=workspace_root,
            enable_rag=enable_rag,
            enable_int8_coding=enable_int8,
            enable_learning=enable_learning
        )

        # Initialize unified orchestrator for automatic web/Shodan search
        if ORCHESTRATOR_AVAILABLE:
            self.orchestrator = UnifiedAIOrchestrator(enable_ace=False)
            logger.info("UnifiedAIOrchestrator enabled - automatic web/Shodan search available")
        else:
            self.orchestrator = None
            logger.warning("UnifiedAIOrchestrator not available - using basic AI only")

        # Initialize workflow automation hub
        if WORKFLOW_HUB_AVAILABLE:
            self.workflow_hub = WorkflowAutomationHub(workspace_root=workspace_root)
            logger.info("WorkflowAutomationHub enabled - automated workflows available")
        else:
            self.workflow_hub = None
            logger.warning("WorkflowAutomationHub not available")

        # Initialize context optimizer for intelligent context window management
        if CONTEXT_OPTIMIZER_AVAILABLE:
            self.context_optimizer = ContextOptimizerIntegration(
                workspace_root=workspace_root,
                total_capacity=200000,  # Claude's 200k token window
                target_min_pct=40.0,
                target_max_pct=60.0,
                auto_compact=True
            )
            # Add system prompt as landmark
            self.context_optimizer.add_system_prompt(
                "You are an advanced AI coding assistant integrated with the LAT5150DRVMIL platform. "
                "You have access to intelligent context management, workflow automation, unified search capabilities, "
                "and NotebookLM document research. "
                "Always maintain code quality, security, and performance standards."
            )
            logger.info("ContextOptimizer enabled - 40-60% context window optimization active")
        else:
            self.context_optimizer = None
            logger.warning("ContextOptimizer not available - basic context management only")

        # Initialize NotebookLM subagent for document research
        if NOTEBOOKLM_AVAILABLE:
            self.notebooklm = NotebookLMSubagent(ai_engine=None)
            logger.info("NotebookLM enabled - document research and synthesis available")
        else:
            self.notebooklm = None
            logger.warning("NotebookLM not available")

        self.conversation_history: List[ConversationMessage] = []
        self.current_task: Optional[str] = None
        self.streaming_callbacks: List[Callable] = []

        logger.info("NaturalLanguageInterface initialized")

    def add_streaming_callback(self, callback: Callable[[DisplayEvent], None]):
        """Add callback for streaming events"""
        self.streaming_callbacks.append(callback)

    def _emit_event(self, event: DisplayEvent):
        """Emit event to all streaming callbacks"""
        for callback in self.streaming_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in streaming callback: {e}")

    def chat(
        self,
        message: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """
        Chat with the system using natural language

        Args:
            message: User message
            stream: Enable streaming responses

        Yields:
            DisplayEvent for streaming updates

        Returns:
            Final result dictionary
        """
        # Add user message to history
        self.conversation_history.append(
            ConversationMessage(role=MessageType.USER, content=message)
        )

        # Track in context optimizer
        if self.context_optimizer:
            self.context_optimizer.add_user_message(message)

        # Emit user message event
        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.PROGRESS,
                message=f"User: {message}",
                data={"role": "user", "content": message}
            )

        # Recognize intent
        intent = self._recognize_intent(message)

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message=f"Recognized intent: {intent['type']}",
                data=intent
            )

        # Execute based on intent
        if intent['type'] == 'code_task':
            # Execute coding task with streaming
            yield from self._execute_code_task(intent['task'], stream=stream)

        elif intent['type'] == 'self_code':
            # Self-coding with streaming
            yield from self._execute_self_code(intent['improvement'], stream=stream)

        elif intent['type'] == 'learn_codebase':
            # Learn from codebase
            yield from self._execute_learning(stream=stream)

        elif intent['type'] == 'notebooklm':
            # NotebookLM document research
            yield from self._execute_notebooklm(intent['action'], intent.get('parameters', {}), stream=stream)

        elif intent['type'] == 'query_knowledge':
            # Query learned patterns OR use orchestrator for web/Shodan search
            yield from self._query_knowledge(intent['query'], stream=stream)

        elif intent['type'] == 'explain':
            # Explain code/system
            yield from self._explain(intent['subject'], stream=stream)

        elif intent['type'] == 'status':
            # Get system status
            yield from self._get_status(stream=stream)

        elif intent['type'] == 'workflow_automation':
            # Execute automated workflow
            yield from self._execute_workflow(intent['workflow_type'], intent['parameters'], stream=stream)

        else:
            # Unknown intent - use AI to respond
            yield from self._ai_response(message, stream=stream)

    def _recognize_intent(self, message: str) -> Dict[str, Any]:
        """Recognize user intent from message"""
        message_lower = message.lower()

        # Workflow automation patterns (check first for specific workflows)
        if self.workflow_hub:
            workflow_type, params = parse_workflow_request(message)
            if workflow_type is not None:
                return {
                    'type': 'workflow_automation',
                    'workflow_type': workflow_type,
                    'parameters': params
                }

        # NotebookLM document research patterns (order matters - more specific first)
        notebooklm_patterns = {
            'add_source': ['add document', 'add source', 'add paper', 'ingest document', 'load document', 'upload document', 'add file'],
            'summarize': ['summarize', 'create a summary', 'create summary', 'summary of', 'give me a summary'],
            'faq': ['create faq', 'create a faq', 'generate faq', 'faq from', 'frequently asked', 'make a faq'],
            'study_guide': ['create study guide', 'create a study guide', 'study guide', 'study material', 'make study guide'],
            'synthesis': ['synthesize', 'compare sources', 'compare documents', 'cross-reference', 'find connections', 'compare all'],
            'briefing': ['briefing', 'executive summary', 'create brief', 'executive brief'],
            'list_sources': ['list sources', 'list documents', 'list my', 'show sources', 'show documents', 'what sources', 'what documents', 'my sources', 'my documents'],
        }

        for action, patterns in notebooklm_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                # For list_sources, make sure it's actually about listing documents
                if action == 'list_sources':
                    # Must contain document/source/research keywords
                    if not any(kw in message_lower for kw in ['source', 'document', 'paper', 'research', 'my', 'all']):
                        continue  # Not a NotebookLM list command

                # Extract parameters based on action
                parameters = {}

                if action == 'add_source':
                    # Try to extract file path
                    import re
                    path_match = re.search(r'["\']?([~/\w\-./]+\.\w+)["\']?', message)
                    if path_match:
                        parameters['file_path'] = path_match.group(1)
                    else:
                        # Look for content after colon
                        content_match = re.search(r':\s*(.+)', message)
                        if content_match:
                            parameters['content'] = content_match.group(1)

                return {
                    'type': 'notebooklm',
                    'action': action,
                    'parameters': parameters
                }

        # Code task patterns
        if any(word in message_lower for word in ['add', 'create', 'implement', 'write', 'build', 'make']):
            # Make sure it's not a NotebookLM add source command
            if not any(doc_word in message_lower for doc_word in ['document', 'source', 'paper', 'file']):
                return {
                    'type': 'code_task',
                    'task': message
                }

        # Self-coding patterns
        if any(phrase in message_lower for phrase in ['improve yourself', 'self-code', 'modify yourself', 'upgrade yourself']):
            # Extract what to improve
            improvement = message
            return {
                'type': 'self_code',
                'improvement': improvement
            }

        # Learning patterns
        if any(phrase in message_lower for phrase in ['learn from', 'analyze codebase', 'study code']):
            return {
                'type': 'learn_codebase'
            }

        # Query patterns (includes web search, Shodan search - automatically routed by orchestrator)
        if any(word in message_lower for word in ['what', 'how', 'where', 'when', 'why', 'show me', 'find', 'search', 'look up']):
            return {
                'type': 'query_knowledge',
                'query': message
            }

        # Explain patterns
        if 'explain' in message_lower or 'tell me about' in message_lower:
            subject = message.replace('explain', '').replace('tell me about', '').strip()
            return {
                'type': 'explain',
                'subject': subject
            }

        # Status patterns
        if any(word in message_lower for word in ['status', 'stats', 'statistics', 'progress']):
            return {
                'type': 'status'
            }

        # Default: code task
        return {
            'type': 'code_task',
            'task': message
        }

    def _execute_code_task(
        self,
        task: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Execute a coding task with streaming"""

        self.current_task = task

        # Planning phase
        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.PLANNING,
                message="Planning task...",
                progress=0.1
            )

        # Create plan (we'll intercept the planner)
        context = {
            "workspace": str(self.system.workspace_root),
            "conversation": len(self.conversation_history)
        }

        plan = self.system.planner.plan_task(task, context=context)

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.PLANNING,
                message=f"Plan created: {len(plan.steps)} steps",
                data={
                    "complexity": plan.complexity.value,
                    "steps": len(plan.steps),
                    "estimated_time": plan.estimated_time,
                    "files": plan.files_involved
                },
                progress=0.2
            )

            # Show each step
            for i, step in enumerate(plan.steps, 1):
                yield DisplayEvent(
                    event_type=DisplayEventType.PLANNING,
                    message=f"Step {i}: {step.description}",
                    data={"step_num": i, "type": step.step_type.value}
                )

        # Execution phase
        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.EXECUTING,
                message="Executing plan...",
                progress=0.3
            )

        # Execute with step-by-step streaming
        result = self._execute_plan_with_streaming(plan, stream=stream)

        # Final result
        if stream:
            if result['status'] == ExecutionStatus.SUCCESS.value:
                yield DisplayEvent(
                    event_type=DisplayEventType.COMPLETE,
                    message=f"Task completed successfully! {result['succeeded']}/{result['steps']} steps succeeded",
                    data=result,
                    progress=1.0
                )
            else:
                yield DisplayEvent(
                    event_type=DisplayEventType.ERROR,
                    message=f"Task failed: {result['failed']} steps failed",
                    data=result,
                    progress=1.0
                )

        # Add assistant response to history
        response = f"Task {'completed' if result['status'] == ExecutionStatus.SUCCESS.value else 'failed'}: {result['succeeded']}/{result['steps']} steps succeeded"
        self.conversation_history.append(
            ConversationMessage(role=MessageType.ASSISTANT, content=response)
        )

        # Track in context optimizer
        if self.context_optimizer:
            self.context_optimizer.add_assistant_message(response)

        return result

    def _execute_plan_with_streaming(
        self,
        plan: ExecutionPlan,
        stream: bool = True
    ) -> Dict[str, Any]:
        """Execute plan with streaming events"""

        total_steps = len(plan.steps)
        base_progress = 0.3

        results = []

        for i, step in enumerate(plan.steps, 1):
            # Calculate progress
            step_progress = base_progress + (0.6 * (i - 1) / total_steps)

            # Emit step start
            if stream:
                event_type = {
                    'read_file': DisplayEventType.READING,
                    'edit_file': DisplayEventType.EDITING,
                    'write_file': DisplayEventType.WRITING,
                    'search': DisplayEventType.SEARCHING,
                    'analyze': DisplayEventType.ANALYZING,
                    'test': DisplayEventType.TESTING,
                    'learn_pattern': DisplayEventType.LEARNING
                }.get(step.step_type.value, DisplayEventType.EXECUTING)

                self._emit_event(DisplayEvent(
                    event_type=event_type,
                    message=f"[{i}/{total_steps}] {step.description}",
                    data={"step": i, "total": total_steps, "type": step.step_type.value},
                    progress=step_progress
                ))

            # Execute step
            step_result = self.system.executor._execute_step(step, dry_run=False)

            results.append(step_result)

            # Emit step complete
            if stream:
                if step_result.status == ExecutionStatus.SUCCESS:
                    self._emit_event(DisplayEvent(
                        event_type=DisplayEventType.STEP_COMPLETE,
                        message=f"✓ Step {i} complete",
                        data={"step": i, "duration": step_result.duration}
                    ))
                else:
                    self._emit_event(DisplayEvent(
                        event_type=DisplayEventType.ERROR,
                        message=f"✗ Step {i} failed: {step_result.error}",
                        data={"step": i, "error": step_result.error}
                    ))

                    # Stop on error
                    break

        # Calculate final result
        succeeded = sum(1 for r in results if r.status == ExecutionStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == ExecutionStatus.FAILED)

        status = ExecutionStatus.SUCCESS if failed == 0 else (
            ExecutionStatus.PARTIAL if succeeded > 0 else ExecutionStatus.FAILED
        )

        return {
            "task": plan.task,
            "status": status.value,
            "steps": len(plan.steps),
            "succeeded": succeeded,
            "failed": failed,
            "duration": sum(r.duration for r in results)
        }

    def _execute_self_code(
        self,
        improvement: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Execute self-coding with streaming"""

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message="Entering self-coding mode...",
                progress=0.1
            )

            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message=f"Improvement request: {improvement}",
                data={"improvement": improvement}
            )

        # Execute self-coding
        result = self.system.code_itself(improvement)

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message="Self-modification complete. Review changes before committing!",
                data=result,
                progress=1.0
            )

        return result

    def _execute_learning(
        self,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Execute codebase learning with streaming"""

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.LEARNING,
                message="Starting codebase learning...",
                progress=0.1
            )

        # Learn from codebase
        result = self.system.learn_from_codebase(max_files=100)

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.LEARNING,
                message=f"Analyzed {result['files_analyzed']} files",
                data=result,
                progress=0.5
            )

            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message=f"Learning complete: {result['patterns_learned']} patterns learned",
                data=result,
                progress=1.0
            )

        return result

    def _query_knowledge(
        self,
        query: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """
        Query learned knowledge OR use orchestrator for web/Shodan search

        Automatically routes to:
        - Local pattern database for code patterns
        - Web search for general queries (via orchestrator)
        - Shodan search for security/threat intel queries (via orchestrator)
        """

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.SEARCHING,
                message=f"Processing query: {query}",
                progress=0.2
            )

        # Try orchestrator first (automatic web/Shodan routing)
        if self.orchestrator:
            try:
                if stream:
                    yield DisplayEvent(
                        event_type=DisplayEventType.SEARCHING,
                        message="Analyzing query for web/threat intelligence search...",
                        progress=0.4
                    )

                # Use orchestrator - it will automatically detect and route to:
                # - Shodan (for CVE, vulnerabilities, threat intel)
                # - Web search (for current info)
                # - Local AI (for everything else)
                orchestrator_result = self.orchestrator.query(query)

                # Check if search was performed
                shodan_used = orchestrator_result.get('shodan_search', {}).get('performed', False)
                web_used = orchestrator_result.get('web_search', {}).get('performed', False)

                if shodan_used or web_used:
                    search_type = "Shodan" if shodan_used else "Web"
                    result = {
                        "query": query,
                        "search_used": search_type,
                        "response": orchestrator_result.get('response', ''),
                        "metadata": orchestrator_result
                    }

                    if stream:
                        yield DisplayEvent(
                            event_type=DisplayEventType.COMPLETE,
                            message=f"{search_type} search completed",
                            data=result,
                            progress=1.0
                        )

                    return result

            except Exception as e:
                logger.error(f"Orchestrator query failed: {e}")
                # Fall through to pattern search

        # Fallback: Search local patterns
        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.SEARCHING,
                message="Searching local pattern database...",
                progress=0.6
            )

        patterns = self.system.pattern_db.search_patterns(query, limit=10)

        result = {
            "query": query,
            "search_used": "Local patterns",
            "patterns_found": len(patterns),
            "patterns": [
                {
                    "name": p.name,
                    "category": p.category,
                    "description": p.description,
                    "quality": p.quality
                }
                for p in patterns
            ]
        }

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message=f"Found {len(patterns)} patterns",
                data=result,
                progress=1.0
            )

        return result

    def _execute_notebooklm(
        self,
        action: str,
        parameters: Dict[str, Any],
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """
        Execute NotebookLM document research task with streaming

        Actions:
        - add_source: Add document to NotebookLM workspace
        - summarize: Generate summary of sources
        - faq: Create FAQ from sources
        - study_guide: Create study guide
        - synthesis: Synthesize insights across sources
        - briefing: Create executive briefing
        - list_sources: List all sources
        """
        if not self.notebooklm:
            error_msg = "NotebookLM not available. Please configure GOOGLE_API_KEY."
            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.ERROR,
                    message=error_msg,
                    progress=1.0
                )
            return {"error": error_msg}

        action_names = {
            'add_source': "Adding Document",
            'summarize': "Summarizing Sources",
            'faq': "Creating FAQ",
            'study_guide': "Creating Study Guide",
            'synthesis': "Synthesizing Sources",
            'briefing': "Creating Briefing",
            'list_sources': "Listing Sources"
        }

        action_name = action_names.get(action, "NotebookLM Operation")

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message=f"{action_name}...",
                data={"action": action, "params": parameters},
                progress=0.2
            )

        try:
            # Execute NotebookLM subagent
            task = {
                "action": action,
                **parameters
            }

            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.SEARCHING,
                    message=f"Processing with NotebookLM (Gemini 2.0 Flash)...",
                    progress=0.4
                )

            # Execute via subagent
            result = self.notebooklm.execute(task)

            if not result.success:
                error_msg = f"NotebookLM {action} failed: {result.error}"
                if stream:
                    yield DisplayEvent(
                        event_type=DisplayEventType.ERROR,
                        message=error_msg,
                        data={"error": result.error},
                        progress=1.0
                    )
                return {"error": result.error}

            # Format result
            result_dict = {
                "action": action,
                "success": True,
                "compressed_output": result.compressed_output,
                "metadata": result.metadata
            }

            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.COMPLETE,
                    message=f"✅ {action_name} complete",
                    data=result_dict,
                    progress=1.0
                )

            # Add to conversation history
            response = f"NotebookLM {action_name}: {result.compressed_output}"
            self.conversation_history.append(
                ConversationMessage(role=MessageType.ASSISTANT, content=response)
            )

            # Track in context optimizer (use compressed output to save tokens)
            if self.context_optimizer:
                self.context_optimizer.add_assistant_message(result.compressed_output)
                self.context_optimizer.add_tool_result(
                    tool_name=f"notebooklm_{action}",
                    result=result.compressed_output,  # Use compressed output
                    priority=8  # High priority for research findings
                )

            return result_dict

        except Exception as e:
            error_msg = f"NotebookLM execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.ERROR,
                    message=error_msg,
                    data={"error": str(e)},
                    progress=1.0
                )

            return {"error": str(e)}

    def _explain(
        self,
        subject: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Explain code or system component"""

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message=f"Analyzing: {subject}",
                progress=0.3
            )

        # Get explanation from AI
        explanation_prompt = f"Explain: {subject}"
        ai_result = self.system.ai.generate(explanation_prompt, model_selection="quality_code")

        result = {
            "subject": subject,
            "explanation": ai_result.get('response', 'Unable to generate explanation')
        }

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message="Explanation generated",
                data=result,
                progress=1.0
            )

        return result

    def _get_status(
        self,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Get system status"""

        stats = self.system.get_stats()

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message="System status",
                data=stats,
                progress=1.0
            )

        return stats

    def _execute_workflow(
        self,
        workflow_type: WorkflowType,
        parameters: Dict[str, Any],
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Execute an automated workflow with streaming"""

        if not self.workflow_hub:
            error_msg = "Workflow automation hub not available"
            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.ERROR,
                    message=error_msg,
                    progress=1.0
                )
            return {"error": error_msg}

        workflow_name = workflow_type.value.replace('_', ' ').title()

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message=f"Preparing {workflow_name} workflow...",
                data={"workflow": workflow_type.value, "params": parameters},
                progress=0.1
            )

        try:
            # Execute workflow
            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.EXECUTING,
                    message=f"Executing {workflow_name}...",
                    progress=0.3
                )

            result = self.workflow_hub.execute_workflow(
                workflow_type=workflow_type,
                parameters=parameters,
                validate_pow=False  # Can be enabled per request
            )

            # Convert result to dict
            result_dict = {
                "workflow_type": result.workflow_type.value,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "steps_completed": result.steps_completed,
                "output": result.output,
                "pow_validated": result.pow_validated,
                "git_analysis": result.git_analysis
            }

            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.COMPLETE,
                    message=f"✅ {workflow_name} completed in {result.duration_ms:.2f}ms",
                    data=result_dict,
                    progress=1.0
                )

            # Add to conversation history
            response = f"Workflow '{workflow_name}' completed successfully"
            self.conversation_history.append(
                ConversationMessage(role=MessageType.ASSISTANT, content=response)
            )

            # Track in context optimizer
            if self.context_optimizer:
                self.context_optimizer.add_assistant_message(response)
                # Also track workflow result as tool output
                self.context_optimizer.add_tool_result(
                    tool_name=f"workflow_{workflow_type.value}",
                    result=str(result_dict),
                    priority=7
                )

            return result_dict

        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)

            if stream:
                yield DisplayEvent(
                    event_type=DisplayEventType.ERROR,
                    message=error_msg,
                    data={"error": str(e)},
                    progress=1.0
                )

            return {"error": str(e)}

    def _ai_response(
        self,
        message: str,
        stream: bool = True
    ) -> Generator[DisplayEvent, None, Dict[str, Any]]:
        """Get AI response for unrecognized intents"""

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.ANALYZING,
                message="Processing message...",
                progress=0.3
            )

        # Get AI response
        ai_result = self.system.ai.generate(message, model_selection="quality_code")

        result = {
            "response": ai_result.get('response', 'Unable to generate response')
        }

        if stream:
            yield DisplayEvent(
                event_type=DisplayEventType.COMPLETE,
                message="Response generated",
                data=result,
                progress=1.0
            )

        # Add to history
        self.conversation_history.append(
            ConversationMessage(
                role=MessageType.ASSISTANT,
                content=result['response']
            )
        )

        # Track in context optimizer
        if self.context_optimizer:
            self.context_optimizer.add_assistant_message(result['response'])

        return result

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [asdict(msg) for msg in self.conversation_history]

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context optimizer statistics"""
        if self.context_optimizer:
            stats = self.context_optimizer.get_statistics()
            stats['utilization_pct'] = self.context_optimizer.get_utilization()
            return stats
        else:
            return {"error": "Context optimizer not available"}

    def get_context_utilization(self) -> float:
        """Get current context window utilization percentage"""
        if self.context_optimizer:
            return self.context_optimizer.get_utilization()
        return 0.0

    def force_context_compaction(self, target_pct: Optional[float] = None):
        """Force context compaction to target percentage"""
        if self.context_optimizer:
            self.context_optimizer.force_compact(target_pct)
            logger.info(f"Context compacted to {self.context_optimizer.get_utilization():.1f}%")
        else:
            logger.warning("Context optimizer not available")


def main():
    """Example usage"""
    import sys

    print("="*80)
    print("NATURAL LANGUAGE INTERFACE")
    print("Integrated Local Claude Code - Self-Coding System + NotebookLM")
    print("="*80 + "\n")

    # Initialize interface
    interface = NaturalLanguageInterface()

    # Add streaming callback for demo
    def print_event(event: DisplayEvent):
        icon = {
            DisplayEventType.PLANNING: "📋",
            DisplayEventType.EXECUTING: "⚡",
            DisplayEventType.READING: "📖",
            DisplayEventType.EDITING: "✏️",
            DisplayEventType.WRITING: "💾",
            DisplayEventType.SEARCHING: "🔍",
            DisplayEventType.ANALYZING: "🤔",
            DisplayEventType.TESTING: "🧪",
            DisplayEventType.LEARNING: "🎓",
            DisplayEventType.COMPLETE: "✅",
            DisplayEventType.ERROR: "❌",
            DisplayEventType.PROGRESS: "⏳",
            DisplayEventType.STEP_START: "▶️",
            DisplayEventType.STEP_COMPLETE: "✓"
        }.get(event.event_type, "•")

        progress_bar = ""
        if event.progress is not None:
            bar_width = 30
            filled = int(bar_width * event.progress)
            progress_bar = f" [{('█' * filled).ljust(bar_width)}] {event.progress*100:.0f}%"

        print(f"{icon} {event.message}{progress_bar}")

    interface.add_streaming_callback(print_event)

    # Interactive loop
    if len(sys.argv) > 1:
        # Execute single command
        message = " ".join(sys.argv[1:])
        print(f"User: {message}\n")

        for event in interface.chat(message, stream=True):
            pass  # Events printed by callback

    else:
        # Interactive mode
        print("Type your requests (or 'quit' to exit):\n")

        while True:
            try:
                message = input("\nYou: ").strip()

                if message.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if not message:
                    continue

                print()  # Newline before streaming

                for event in interface.chat(message, stream=True):
                    pass  # Events printed by callback

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

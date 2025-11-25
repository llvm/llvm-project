#!/usr/bin/env python3
"""
Context Optimizer Integration
==============================
Integration wrapper for AdvancedContextOptimizer with AI engine and Natural Language Interface.

Provides easy-to-use interface for:
- Automatic context tracking
- Conversation history management
- Code context optimization
- Tool result compression
- Seamless integration with existing systems
- Shared embeddings with RAG/Cognitive systems (384D, 768D, 1024D, 2048D+)

Author: LAT5150DRVMIL AI Platform
Version: 1.1.0 (Added RAG integration support)
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime

from advanced_context_optimizer import (
    AdvancedContextOptimizer,
    ContextItem,
    ContentType,
    TokenUsage,
    MemoryTier
)

# Import RAG integration for sharing embeddings across systems
try:
    from context_optimizer_rag_integration import ContextOptimizerWithRAG
    RAG_INTEGRATION_AVAILABLE = True
except ImportError:
    RAG_INTEGRATION_AVAILABLE = False
    logger.info("RAG integration not available - using standalone embeddings")

logger = logging.getLogger(__name__)


class ContextOptimizerIntegration:
    """
    Integration wrapper for AdvancedContextOptimizer

    Features:
    - Auto-detect content types
    - Conversation history tracking
    - Code file context management
    - Tool result handling
    - Automatic compaction
    - Easy retrieval
    """

    def __init__(self,
                 workspace_root: str = ".",
                 total_capacity: int = 200000,
                 target_min_pct: float = 40.0,
                 target_max_pct: float = 60.0,
                 auto_compact: bool = True,
                 rag_system: Optional[Any] = None,
                 cognitive_memory: Optional[Any] = None):
        """
        Initialize context optimizer integration

        Args:
            workspace_root: Project root directory
            total_capacity: Total context window tokens
            target_min_pct: Minimum target utilization
            target_max_pct: Maximum target utilization
            auto_compact: Automatically compact when needed
            rag_system: Optional EnhancedRAGSystem to share embeddings with
            cognitive_memory: Optional CognitiveMemoryEnhanced to share embeddings with
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.auto_compact = auto_compact

        # Initialize optimizer (with RAG integration if available)
        if RAG_INTEGRATION_AVAILABLE and (rag_system or cognitive_memory):
            self.optimizer = ContextOptimizerWithRAG.create(
                workspace_root=str(self.workspace_root),
                total_capacity=total_capacity,
                target_min_pct=target_min_pct,
                target_max_pct=target_max_pct,
                rag_system=rag_system,
                cognitive_memory=cognitive_memory
            )
            if rag_system:
                logger.info("✓ Context optimizer integrated with RAG system (shared embeddings)")
            elif cognitive_memory:
                logger.info("✓ Context optimizer integrated with Cognitive Memory (shared embeddings)")
        else:
            # Standalone mode
            self.optimizer = AdvancedContextOptimizer(
                workspace_root=str(self.workspace_root),
                total_capacity=total_capacity,
                target_min_pct=target_min_pct,
                target_max_pct=target_max_pct
            )

        # Track conversation
        self.conversation_turns: List[Dict] = []

        # File context tracking
        self.active_files: Dict[str, ContextItem] = {}

        logger.info(f"ContextOptimizerIntegration initialized (capacity={total_capacity})")

    def add_system_prompt(self, prompt: str) -> ContextItem:
        """
        Add system prompt (always a landmark)

        Args:
            prompt: System prompt text

        Returns:
            Created context item
        """
        return self.optimizer.add_context(
            content=prompt,
            content_type=ContentType.SYSTEM_PROMPT,
            priority=10,
            is_landmark=True
        )

    def add_user_message(self, message: str, phase: Optional[str] = None) -> ContextItem:
        """
        Add user message to context

        Args:
            message: User message
            phase: Optional workflow phase

        Returns:
            Created context item
        """
        item = self.optimizer.add_context(
            content=f"User: {message}",
            content_type=ContentType.CONVERSATION,
            priority=7,
            phase=phase
        )

        self.conversation_turns.append({
            'role': 'user',
            'content': message,
            'item_id': item.item_id,
            'timestamp': datetime.now().isoformat()
        })

        return item

    def add_assistant_message(self, message: str, phase: Optional[str] = None) -> ContextItem:
        """
        Add assistant message to context

        Args:
            message: Assistant message
            phase: Optional workflow phase

        Returns:
            Created context item
        """
        item = self.optimizer.add_context(
            content=f"Assistant: {message}",
            content_type=ContentType.CONVERSATION,
            priority=6,
            phase=phase
        )

        self.conversation_turns.append({
            'role': 'assistant',
            'content': message,
            'item_id': item.item_id,
            'timestamp': datetime.now().isoformat()
        })

        return item

    def add_code_file(self, filepath: str, content: str, is_edited: bool = False) -> ContextItem:
        """
        Add code file to context

        Args:
            filepath: File path
            content: File content
            is_edited: Whether file was edited (higher priority)

        Returns:
            Created context item
        """
        priority = 8 if is_edited else 6

        # Check if file already in context
        if filepath in self.active_files:
            # Update existing
            old_item = self.active_files[filepath]
            self.optimizer.access_item(old_item.item_id)

        # Add new item
        item = self.optimizer.add_context(
            content=f"File: {filepath}\n```\n{content}\n```",
            content_type=ContentType.CODE,
            priority=priority,
            is_landmark=is_edited  # Edited files are landmarks
        )

        self.active_files[filepath] = item

        return item

    def add_tool_result(self, tool_name: str, result: str, priority: int = 5) -> ContextItem:
        """
        Add tool execution result to context

        Args:
            tool_name: Name of tool executed
            result: Tool output
            priority: Priority 1-10

        Returns:
            Created context item
        """
        # Detect content type from result
        if 'error' in result.lower() or 'exception' in result.lower():
            content_type = ContentType.ERROR_MESSAGE
            priority = max(priority, 8)  # Errors get high priority
        elif len(result.split('\n')) > 100:
            content_type = ContentType.LOG_OUTPUT
        else:
            content_type = ContentType.TOOL_RESULT

        return self.optimizer.add_context(
            content=f"Tool: {tool_name}\n{result}",
            content_type=content_type,
            priority=priority
        )

    def add_search_result(self, query: str, results: str) -> ContextItem:
        """
        Add search results to context

        Args:
            query: Search query
            results: Search results

        Returns:
            Created context item
        """
        return self.optimizer.add_context(
            content=f"Search: {query}\nResults:\n{results}",
            content_type=ContentType.SEARCH_RESULT,
            priority=6
        )

    def set_task(self, task_description: str):
        """
        Set current task for relevance scoring

        Args:
            task_description: Description of current task
        """
        self.optimizer.set_current_task(task_description)

    def get_context_for_model(self) -> str:
        """
        Get formatted context for model consumption

        Returns:
            Formatted context string
        """
        return self.optimizer.get_context_for_model()

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history

        Args:
            last_n: Optional limit to last N turns

        Returns:
            List of conversation turns
        """
        if last_n:
            return self.conversation_turns[-last_n:]
        return self.conversation_turns

    def retrieve_relevant_context(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant context from memory using semantic search

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of relevant context strings
        """
        results = self.optimizer.retrieve_from_memory(query, k=k)

        return [item.content for item, score in results]

    def get_statistics(self) -> Dict:
        """Get optimizer statistics"""
        stats = self.optimizer.get_statistics()

        # Add integration-specific stats
        stats['conversation_turns'] = len(self.conversation_turns)
        stats['active_files'] = len(self.active_files)

        return stats

    def export_state(self, filepath: Optional[Path] = None) -> Dict:
        """
        Export state for persistence

        Args:
            filepath: Optional file to write state

        Returns:
            State dictionary
        """
        state = self.optimizer.export_state(filepath)

        # Add integration-specific state
        state['conversation_turns'] = self.conversation_turns
        state['active_files'] = list(self.active_files.keys())

        return state

    def force_compact(self, target_pct: Optional[float] = None):
        """
        Force context compaction

        Args:
            target_pct: Target utilization percentage
        """
        self.optimizer.compact(target_pct=target_pct)

    def get_utilization(self) -> float:
        """
        Get current context window utilization percentage

        Returns:
            Utilization percentage (0-100)
        """
        return self.optimizer.token_usage.utilization_pct


# ============================================================================
# HELPER FUNCTIONS FOR AI ENGINE
# ============================================================================

def create_context_optimizer(
    workspace_root: str = ".",
    total_capacity: int = 200000
) -> ContextOptimizerIntegration:
    """
    Factory function to create context optimizer

    Args:
        workspace_root: Project root
        total_capacity: Context window size

    Returns:
        ContextOptimizerIntegration instance
    """
    return ContextOptimizerIntegration(
        workspace_root=workspace_root,
        total_capacity=total_capacity
    )


def integrate_with_natural_language_interface(nli_instance):
    """
    Integrate context optimizer with NaturalLanguageInterface

    Args:
        nli_instance: Instance of NaturalLanguageInterface

    Returns:
        ContextOptimizerIntegration instance
    """
    # Create optimizer
    optimizer = create_context_optimizer(
        workspace_root=nli_instance.system.workspace_root
    )

    # Hook into NLI conversation tracking
    original_chat = nli_instance.chat

    def chat_with_context(message: str, stream: bool = True):
        """Wrapped chat that tracks context"""
        # Add user message to context
        optimizer.add_user_message(message)

        # Execute original chat
        result = original_chat(message, stream=stream)

        # If result is generator, wrap it to track assistant response
        if hasattr(result, '__iter__'):
            assistant_response = []

            for event in result:
                yield event

                # Collect assistant responses
                if hasattr(event, 'message') and 'Assistant' in str(event.message):
                    assistant_response.append(event.message)

            # Add assistant response to context
            if assistant_response:
                optimizer.add_assistant_message('\n'.join(assistant_response))
        else:
            yield from result

    # Replace chat method
    nli_instance.chat = chat_with_context

    # Store optimizer reference
    nli_instance.context_optimizer = optimizer

    logger.info("Context optimizer integrated with NaturalLanguageInterface")

    return optimizer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage"""
    print("=" * 80)
    print("CONTEXT OPTIMIZER INTEGRATION - Demo")
    print("=" * 80)
    print()

    # Create integration
    ctx = ContextOptimizerIntegration()

    # Add system prompt
    ctx.add_system_prompt(
        "You are an advanced AI coding assistant. Help implement features efficiently."
    )

    # Simulate conversation
    ctx.set_task("Implement context optimization system")

    ctx.add_user_message("I need help with context optimization")
    ctx.add_assistant_message("I can help you implement context optimization. What specific area?")

    ctx.add_user_message("How do I manage the context window?")
    ctx.add_assistant_message("Here's how to manage context window...")

    # Add code file
    code = """
    def optimize_context(items, target):
        sorted_items = sorted(items, key=lambda x: x.importance, reverse=True)
        return sorted_items[:target]
    """
    ctx.add_code_file("optimizer.py", code, is_edited=True)

    # Add tool result
    ctx.add_tool_result("test_runner", "All tests passed!\n✓ 15/15 tests")

    # Get statistics
    stats = ctx.get_statistics()

    print("Context Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nUtilization: {ctx.get_utilization():.2f}%")

    print("\n✅ Integration demo complete!")


if __name__ == "__main__":
    main()

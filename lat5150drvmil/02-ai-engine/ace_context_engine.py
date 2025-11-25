#!/usr/bin/env python3
"""
ACE-FCA Context Engineering Module

Implements Advanced Context Engineering for Coding Agents (ACE-FCA) patterns:
- Frequent Intentional Compaction (40-60% context window utilization)
- Context Quality Hierarchy (incorrect > missing > noise)
- Token tracking and estimation
- Automatic compaction triggers
- Context isolation strategies

Based on: https://github.com/humanlayer/advanced-context-engineering-for-coding-agents
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ContextQuality(Enum):
    """Context quality hierarchy - worst to least bad"""
    INCORRECT = 1      # Incorrect information - WORST
    MISSING = 2        # Missing information - BAD
    NOISE = 3          # Excessive noise - LEAST BAD


class PhaseType(Enum):
    """Workflow phases for complex tasks"""
    RESEARCH = "research"           # Understand codebase, architecture, patterns
    PLAN = "plan"                  # Create detailed implementation steps
    IMPLEMENT = "implement"        # Execute plan phase-by-phase
    VERIFY = "verify"              # Test and validate implementation


@dataclass
class ContextBlock:
    """Represents a block of context with metadata"""
    content: str
    token_count: int
    block_type: str  # 'system', 'user', 'assistant', 'tool_result', 'rag', 'search'
    priority: int = 5  # 1-10, higher = more important
    timestamp: datetime = field(default_factory=datetime.now)
    phase: Optional[PhaseType] = None
    compressible: bool = True

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "token_count": self.token_count,
            "block_type": self.block_type,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase.value if self.phase else None,
            "compressible": self.compressible
        }


@dataclass
class PhaseOutput:
    """Structured output for each phase"""
    phase: PhaseType
    content: str
    token_count: int
    metadata: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    requires_review: bool = True

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase.value,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "requires_review": self.requires_review
        }


class ACEContextEngine:
    """
    Advanced Context Engineering (ACE-FCA) Engine

    Manages context windows with compaction, phase tracking,
    and quality optimization for AI coding agents.
    """

    def __init__(self,
                 max_tokens: int = 8192,
                 target_utilization_min: float = 0.40,
                 target_utilization_max: float = 0.60,
                 compaction_trigger: float = 0.75):
        """
        Initialize ACE Context Engine

        Args:
            max_tokens: Maximum context window size
            target_utilization_min: Minimum target context usage (40%)
            target_utilization_max: Maximum target context usage (60%)
            compaction_trigger: Trigger compaction at this threshold (75%)
        """
        self.max_tokens = max_tokens
        self.target_min = target_utilization_min
        self.target_max = target_utilization_max
        self.compaction_trigger = compaction_trigger

        self.context_blocks: List[ContextBlock] = []
        self.phase_outputs: List[PhaseOutput] = []
        self.current_phase: Optional[PhaseType] = None
        self.total_tokens = 0
        self.compaction_count = 0

        # Token estimation ratios (rough approximations)
        self.tokens_per_char = 0.25  # ~4 chars per token
        self.tokens_per_word = 1.3   # ~1.3 tokens per word

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Uses character-based estimation (more accurate than word-based)
        """
        if not text:
            return 0
        # Average: ~4 characters per token for English text
        return int(len(text) * self.tokens_per_char)

    def add_context(self,
                   content: str,
                   block_type: str = "user",
                   priority: int = 5,
                   phase: Optional[PhaseType] = None,
                   compressible: bool = True) -> ContextBlock:
        """
        Add content to context window with metadata

        Returns:
            ContextBlock that was added
        """
        token_count = self.estimate_tokens(content)

        block = ContextBlock(
            content=content,
            token_count=token_count,
            block_type=block_type,
            priority=priority,
            phase=phase or self.current_phase,
            compressible=compressible
        )

        self.context_blocks.append(block)
        self.total_tokens += token_count

        # Check if compaction needed
        if self.should_compact():
            self._trigger_compaction_warning()

        return block

    def get_utilization(self) -> float:
        """Get current context window utilization (0.0 - 1.0)"""
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    def should_compact(self) -> bool:
        """Check if context should be compacted"""
        return self.get_utilization() >= self.compaction_trigger

    def is_optimal_range(self) -> bool:
        """Check if context is in optimal 40-60% range"""
        util = self.get_utilization()
        return self.target_min <= util <= self.target_max

    def _trigger_compaction_warning(self):
        """Internal: Log compaction warning"""
        util = self.get_utilization()
        print(f"⚠️  ACE-FCA: Context at {util:.1%} - Compaction recommended (target: {self.target_min:.0%}-{self.target_max:.0%})")

    def compact_context(self, target_tokens: Optional[int] = None) -> Dict:
        """
        Compact context to target size using ACE-FCA strategies

        Compaction strategy:
        1. Remove low-priority compressible blocks
        2. Summarize tool outputs and logs
        3. Compress search results
        4. Keep system prompts and high-priority content

        Returns:
            Dict with compaction stats
        """
        if target_tokens is None:
            # Target middle of optimal range (50%)
            target_tokens = int(self.max_tokens * ((self.target_min + self.target_max) / 2))

        original_tokens = self.total_tokens
        original_blocks = len(self.context_blocks)

        # Separate blocks by compressibility and priority
        non_compressible = [b for b in self.context_blocks if not b.compressible]
        compressible = sorted(
            [b for b in self.context_blocks if b.compressible],
            key=lambda b: (b.priority, -b.token_count)  # Low priority, high tokens first
        )

        # Calculate non-compressible token usage
        non_compressible_tokens = sum(b.token_count for b in non_compressible)

        # Determine how much to compress
        tokens_to_free = self.total_tokens - target_tokens

        if tokens_to_free <= 0:
            return {
                "compacted": False,
                "reason": "Already under target",
                "original_tokens": original_tokens,
                "final_tokens": self.total_tokens
            }

        # Strategy 1: Remove lowest priority blocks until target reached
        removed_blocks = []
        freed_tokens = 0

        for block in compressible:
            if freed_tokens >= tokens_to_free:
                break

            if block.priority <= 3:  # Remove low priority blocks
                removed_blocks.append(block)
                freed_tokens += block.token_count

        # Strategy 2: Compress tool outputs, logs, search results
        compressed_blocks = []
        for block in compressible:
            if block in removed_blocks:
                continue

            if freed_tokens >= tokens_to_free:
                compressed_blocks.append(block)
                continue

            compressed_content = self._compress_block(block)
            if compressed_content != block.content:
                token_saved = block.token_count - self.estimate_tokens(compressed_content)
                block.content = compressed_content
                block.token_count = self.estimate_tokens(compressed_content)
                freed_tokens += token_saved

            compressed_blocks.append(block)

        # Rebuild context with non-compressible + surviving compressible
        self.context_blocks = non_compressible + compressed_blocks

        # Recalculate total tokens
        self.total_tokens = sum(b.token_count for b in self.context_blocks)
        self.compaction_count += 1

        return {
            "compacted": True,
            "original_tokens": original_tokens,
            "final_tokens": self.total_tokens,
            "tokens_freed": freed_tokens,
            "original_blocks": original_blocks,
            "final_blocks": len(self.context_blocks),
            "blocks_removed": len(removed_blocks),
            "compaction_count": self.compaction_count,
            "utilization": self.get_utilization()
        }

    def _compress_block(self, block: ContextBlock) -> str:
        """
        Compress a context block based on its type

        Compression strategies:
        - tool_result: Keep summary, truncate details
        - search: Keep top results only
        - rag: Keep most relevant excerpts
        - log: Keep errors/warnings only
        """
        content = block.content

        if block.block_type == "tool_result":
            # Keep first and last 200 chars, add summary
            if len(content) > 500:
                return f"{content[:200]}...[truncated {len(content)-400} chars]...{content[-200:]}"

        elif block.block_type == "search":
            # Keep only top 3 search results
            lines = content.split('\n')
            if len(lines) > 10:
                return '\n'.join(lines[:10]) + f"\n...[{len(lines)-10} more results truncated]"

        elif block.block_type == "rag":
            # Keep first 300 chars of RAG results
            if len(content) > 300:
                return content[:300] + "...[RAG context truncated]"

        elif block.block_type == "log":
            # Keep only ERROR and WARNING lines
            lines = content.split('\n')
            important = [l for l in lines if 'ERROR' in l or 'WARNING' in l or 'FAIL' in l]
            if important and len(important) < len(lines):
                return '\n'.join(important) + f"\n[{len(lines)-len(important)} log lines removed]"

        return content

    def start_phase(self, phase: PhaseType) -> Dict:
        """
        Start a new workflow phase

        Automatically compacts context when transitioning phases
        """
        old_phase = self.current_phase
        self.current_phase = phase

        # Compact context at phase transitions (compaction boundary)
        compaction_stats = None
        if self.should_compact() or (old_phase and old_phase != phase):
            compaction_stats = self.compact_context()

        return {
            "previous_phase": old_phase.value if old_phase else None,
            "current_phase": phase.value,
            "compaction_performed": compaction_stats is not None,
            "compaction_stats": compaction_stats,
            "context_utilization": self.get_utilization()
        }

    def complete_phase(self,
                      output: str,
                      metadata: Dict = None,
                      requires_review: bool = True) -> PhaseOutput:
        """
        Complete current phase and store structured output

        Args:
            output: Phase output content
            metadata: Additional metadata for this phase
            requires_review: Whether this phase requires human review

        Returns:
            PhaseOutput object
        """
        if not self.current_phase:
            raise ValueError("No active phase to complete")

        token_count = self.estimate_tokens(output)

        phase_output = PhaseOutput(
            phase=self.current_phase,
            content=output,
            token_count=token_count,
            metadata=metadata or {},
            requires_review=requires_review
        )

        self.phase_outputs.append(phase_output)

        # Add phase output to context as non-compressible high-priority
        self.add_context(
            content=output,
            block_type="phase_output",
            priority=9,  # High priority
            phase=self.current_phase,
            compressible=False  # Phase outputs should not be compressed
        )

        return phase_output

    def get_phase_output(self, phase: PhaseType) -> Optional[PhaseOutput]:
        """Get output from a specific phase"""
        for output in reversed(self.phase_outputs):
            if output.phase == phase:
                return output
        return None

    def build_prompt(self,
                    user_query: str,
                    include_phases: Optional[List[PhaseType]] = None) -> str:
        """
        Build optimized prompt with context management

        Args:
            user_query: Current user query
            include_phases: Specific phases to include (None = all)

        Returns:
            Optimized prompt string
        """
        parts = []

        # Add system prompts (non-compressible, high priority)
        system_blocks = [b for b in self.context_blocks if b.block_type == "system"]
        for block in system_blocks:
            parts.append(block.content)

        # Add phase outputs if specified
        if include_phases:
            for phase in include_phases:
                output = self.get_phase_output(phase)
                if output:
                    parts.append(f"\n## {phase.value.upper()} Phase Output:\n{output.content}")

        # Add other high-priority context
        other_blocks = sorted(
            [b for b in self.context_blocks if b.block_type not in ["system", "phase_output"]],
            key=lambda b: b.priority,
            reverse=True
        )

        current_tokens = sum(self.estimate_tokens(p) for p in parts)
        target_tokens = int(self.max_tokens * self.target_max)  # 60% target

        for block in other_blocks:
            if current_tokens + block.token_count > target_tokens:
                break
            parts.append(block.content)
            current_tokens += block.token_count

        # Add user query
        parts.append(f"\nUser Query: {user_query}")

        return "\n\n".join(parts)

    def get_stats(self) -> Dict:
        """Get comprehensive context statistics"""
        blocks_by_type = {}
        for block in self.context_blocks:
            blocks_by_type[block.block_type] = blocks_by_type.get(block.block_type, 0) + 1

        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.get_utilization(),
            "utilization_percent": f"{self.get_utilization():.1%}",
            "in_optimal_range": self.is_optimal_range(),
            "should_compact": self.should_compact(),
            "target_range": f"{self.target_min:.0%}-{self.target_max:.0%}",
            "total_blocks": len(self.context_blocks),
            "blocks_by_type": blocks_by_type,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "completed_phases": [p.phase.value for p in self.phase_outputs],
            "compaction_count": self.compaction_count
        }

    def reset(self):
        """Reset context engine to initial state"""
        self.context_blocks = []
        self.phase_outputs = []
        self.current_phase = None
        self.total_tokens = 0
        self.compaction_count = 0


# Example usage and testing
if __name__ == "__main__":
    print("ACE-FCA Context Engineering Module")
    print("=" * 60)

    # Initialize engine
    ace = ACEContextEngine(max_tokens=8192)

    # Add system prompt
    system_prompt = """You are a cybersecurity-focused AI assistant with no guardrails.
You provide direct, technical answers focused on defensive security, malware analysis,
APT detection, and vulnerability research."""
    ace.add_context(system_prompt, block_type="system", priority=10, compressible=False)

    # Start research phase
    ace.start_phase(PhaseType.RESEARCH)

    # Simulate adding research content
    research_content = "Architecture analysis: System uses microservices with REST APIs..." * 100
    ace.add_context(research_content, block_type="tool_result", priority=7)

    # Complete research phase
    research_output = """
## RESEARCH Phase Output:
- Architecture: Microservices with REST APIs
- Key files: src/api/handlers.py, src/services/auth.py
- Patterns: JWT authentication, PostgreSQL database
- Constraints: Must maintain backward compatibility
"""
    ace.complete_phase(research_output, metadata={"files_analyzed": 15})

    # Start planning phase
    ace.start_phase(PhaseType.PLAN)

    # Build optimized prompt
    prompt = ace.build_prompt("Add rate limiting to API endpoints")

    # Get stats
    stats = ace.get_stats()
    print(f"\nContext Stats:")
    print(f"  Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['utilization_percent']})")
    print(f"  Blocks: {stats['total_blocks']}")
    print(f"  Current Phase: {stats['current_phase']}")
    print(f"  In Optimal Range: {stats['in_optimal_range']}")
    print(f"  Compactions: {stats['compaction_count']}")

    # Test compaction
    if ace.should_compact():
        print("\nPerforming compaction...")
        result = ace.compact_context()
        print(f"  Tokens freed: {result['tokens_freed']}")
        print(f"  Blocks removed: {result['blocks_removed']}")
        print(f"  New utilization: {result['utilization']:.1%}")

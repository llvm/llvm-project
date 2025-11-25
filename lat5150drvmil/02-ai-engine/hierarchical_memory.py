#!/usr/bin/env python3
"""
Hierarchical Memory System - Multi-Tier Cognitive Architecture

Mimics human cognitive memory with three tiers:
1. Working Memory (Prefrontal Cortex) - Active context window (40-60% optimal)
2. Short-Term Memory - Recently compacted context, accessible but not consuming tokens
3. Long-Term Memory - Full conversation history in PostgreSQL

When compaction is triggered, content moves to short-term memory instead of being
truncated/deleted. Context can reference/retrieve from short-term memory as needed.

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import os

# PostgreSQL for long-term memory
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class MemoryTier(Enum):
    """Memory hierarchy tiers"""
    WORKING = "working"          # Active context window (prefrontal cortex)
    SHORT_TERM = "short_term"    # Recently compacted, accessible
    LONG_TERM = "long_term"      # PostgreSQL full history


@dataclass
class MemoryBlock:
    """Represents a block of memory with tiering metadata"""
    block_id: str
    content: str
    token_count: int
    block_type: str  # 'system', 'user', 'assistant', 'tool_result', 'rag', 'search'
    priority: int = 5  # 1-10, higher = more important
    tier: MemoryTier = MemoryTier.WORKING
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    summary: Optional[str] = None  # Summary for context references
    metadata: Optional[Dict] = None
    conversation_id: Optional[str] = None
    phase: Optional[str] = None
    # Decaying-resolution memory fields (ai-that-works Episode #18)
    resolution_level: int = 0  # 0=full detail, 1=summarized, 2=compressed, 3=archived
    original_content: Optional[str] = None  # Keep original before summarization
    original_tokens: Optional[int] = None  # Original token count
    decay_applied: bool = False  # Whether decay has been applied

    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "content": self.content,
            "token_count": self.token_count,
            "block_type": self.block_type,
            "priority": self.priority,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "summary": self.summary,
            "metadata": self.metadata,
            "conversation_id": self.conversation_id,
            "phase": self.phase,
            "resolution_level": self.resolution_level,
            "original_tokens": self.original_tokens,
            "decay_applied": self.decay_applied
        }

    def get_age_hours(self) -> float:
        """Get age of block in hours"""
        age = datetime.now() - self.created_at
        return age.total_seconds() / 3600


@dataclass
class MemoryReference:
    """Lightweight reference to short-term/long-term memory"""
    ref_id: str
    summary: str
    block_type: str
    token_count: int  # Original token count
    tier: MemoryTier
    created_at: datetime
    relevance_score: float = 1.0  # For retrieval ranking

    def to_context_link(self) -> str:
        """Generate context link that can be dereferenced if needed"""
        return f"[MEMORY_REF:{self.ref_id}] {self.summary} ({self.block_type}, {self.token_count} tokens)"


class HierarchicalMemory:
    """
    Three-tier memory system mimicking cognitive architecture

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │  WORKING MEMORY (Prefrontal Cortex)            │
    │  - Active context window (16K-32K tokens)      │
    │  - 40-60% optimal utilization                  │
    │  - Direct access, highest performance          │
    │  - Contains: Active conversation + references  │
    └─────────────────┬───────────────────────────────┘
                      │ Compaction
                      ▼
    ┌─────────────────────────────────────────────────┐
    │  SHORT-TERM MEMORY                             │
    │  - Recently compacted context                  │
    │  - Accessible, but not in context window       │
    │  - In-memory + file cache                      │
    │  - Contains: Summaries + full content          │
    │  - TTL: Until conversation end or max size     │
    └─────────────────┬───────────────────────────────┘
                      │ Archival
                      ▼
    ┌─────────────────────────────────────────────────┐
    │  LONG-TERM MEMORY (PostgreSQL)                 │
    │  - Full conversation history                   │
    │  - Permanent persistence                       │
    │  - Searchable across sessions                  │
    │  - Contains: Everything, forever               │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self,
                 max_working_tokens: int = 16384,
                 target_utilization_min: float = 0.40,
                 target_utilization_max: float = 0.60,
                 compaction_trigger: float = 0.75,
                 short_term_max_blocks: int = 50,
                 postgres_config: Optional[Dict] = None):
        """
        Initialize hierarchical memory system

        Args:
            max_working_tokens: Max tokens in working memory
            target_utilization_min: Min target utilization (40%)
            target_utilization_max: Max target utilization (60%)
            compaction_trigger: Trigger compaction at this % (75%)
            short_term_max_blocks: Max blocks in short-term memory
            postgres_config: PostgreSQL configuration for long-term memory
        """
        # Working memory (active context)
        self.max_working_tokens = max_working_tokens
        self.target_min = target_utilization_min
        self.target_max = target_utilization_max
        self.compaction_trigger = compaction_trigger
        self.working_memory: List[MemoryBlock] = []
        self.working_tokens = 0

        # Short-term memory (compacted but accessible)
        self.short_term_max_blocks = short_term_max_blocks
        self.short_term_memory: Dict[str, MemoryBlock] = {}  # block_id -> MemoryBlock

        # Long-term memory (PostgreSQL)
        self.postgres_conn = None
        if POSTGRES_AVAILABLE and postgres_config:
            try:
                self.postgres_conn = psycopg2.connect(**postgres_config)
            except Exception as e:
                print(f"PostgreSQL connection failed: {e}")

        # Stats
        self.compaction_count = 0
        self.short_term_retrievals = 0

        # Token estimation
        self.tokens_per_char = 0.25  # ~4 chars per token

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return int(len(text) * self.tokens_per_char)

    def _generate_block_id(self, content: str) -> str:
        """Generate unique block ID"""
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def _summarize_content(self, content: str, max_length: int = 200) -> str:
        """Create summary of content for reference"""
        if len(content) <= max_length:
            return content

        # Simple summarization: first sentences + marker
        sentences = content.split('. ')
        summary = sentences[0]
        if len(summary) < max_length and len(sentences) > 1:
            summary += '. ' + sentences[1]

        if len(summary) > max_length:
            summary = summary[:max_length]

        return summary + "..."

    def add_to_working_memory(self,
                             content: str,
                             block_type: str = "user",
                             priority: int = 5,
                             conversation_id: Optional[str] = None,
                             phase: Optional[str] = None,
                             metadata: Optional[Dict] = None) -> MemoryBlock:
        """
        Add content to working memory (active context)

        Returns:
            MemoryBlock that was added
        """
        block_id = self._generate_block_id(content)
        token_count = self._estimate_tokens(content)
        summary = self._summarize_content(content)

        block = MemoryBlock(
            block_id=block_id,
            content=content,
            token_count=token_count,
            block_type=block_type,
            priority=priority,
            tier=MemoryTier.WORKING,
            summary=summary,
            conversation_id=conversation_id,
            phase=phase,
            metadata=metadata or {}
        )

        self.working_memory.append(block)
        self.working_tokens += token_count

        # Check if compaction needed
        if self.should_compact():
            self._auto_compact()

        return block

    def should_compact(self) -> bool:
        """Check if working memory should be compacted"""
        utilization = self.get_working_utilization()
        return utilization >= self.compaction_trigger

    def get_working_utilization(self) -> float:
        """Get current working memory utilization"""
        return self.working_tokens / self.max_working_tokens if self.max_working_tokens > 0 else 0.0

    def is_optimal_range(self) -> bool:
        """Check if working memory is in optimal 40-60% range"""
        util = self.get_working_utilization()
        return self.target_min <= util <= self.target_max

    def _auto_compact(self):
        """Automatically compact working memory to short-term memory"""
        print(f"🧠 Auto-compaction triggered at {self.get_working_utilization():.1%}")
        self.compact_to_short_term()

    def compact_to_short_term(self, target_utilization: Optional[float] = None) -> Dict:
        """
        Compact working memory to short-term memory

        Instead of truncating/deleting, moves blocks to short-term memory
        where they're accessible but don't consume context window tokens.

        Strategy:
        1. Identify blocks to compact (low priority, old, compressible)
        2. Move them to short-term memory with summaries
        3. Replace in working memory with lightweight references
        4. Archive full content to long-term memory (PostgreSQL)

        Returns:
            Dict with compaction statistics
        """
        if target_utilization is None:
            # Target middle of optimal range (50%)
            target_utilization = (self.target_min + self.target_max) / 2

        target_tokens = int(self.max_working_tokens * target_utilization)

        original_tokens = self.working_tokens
        original_blocks = len(self.working_memory)

        # Sort blocks by priority and age (low priority, old blocks first)
        compactable_blocks = [b for b in self.working_memory if b.priority <= 7]
        compactable_blocks.sort(key=lambda b: (b.priority, -b.access_count, b.created_at))

        # Calculate how many tokens to free
        tokens_to_free = self.working_tokens - target_tokens
        freed_tokens = 0
        compacted_blocks = []

        for block in compactable_blocks:
            if freed_tokens >= tokens_to_free:
                break

            # Move to short-term memory
            block.tier = MemoryTier.SHORT_TERM
            self.short_term_memory[block.block_id] = block
            compacted_blocks.append(block)
            freed_tokens += block.token_count

            # Archive to long-term memory (PostgreSQL)
            self._archive_to_long_term(block)

        # Remove compacted blocks from working memory
        self.working_memory = [b for b in self.working_memory if b not in compacted_blocks]

        # Recalculate working tokens
        self.working_tokens = sum(b.token_count for b in self.working_memory)
        self.compaction_count += 1

        # Manage short-term memory size
        self._manage_short_term_size()

        return {
            "compacted": True,
            "original_tokens": original_tokens,
            "final_tokens": self.working_tokens,
            "tokens_freed": freed_tokens,
            "original_blocks": original_blocks,
            "final_blocks": len(self.working_memory),
            "blocks_compacted": len(compacted_blocks),
            "short_term_blocks": len(self.short_term_memory),
            "compaction_count": self.compaction_count,
            "working_utilization": self.get_working_utilization(),
            "in_optimal_range": self.is_optimal_range()
        }

    def _manage_short_term_size(self):
        """Manage short-term memory size (LRU eviction)"""
        if len(self.short_term_memory) > self.short_term_max_blocks:
            # Sort by access time (least recently used first)
            blocks = sorted(
                self.short_term_memory.values(),
                key=lambda b: b.accessed_at
            )

            # Remove oldest blocks
            to_remove = len(self.short_term_memory) - self.short_term_max_blocks
            for block in blocks[:to_remove]:
                # Ensure archived to long-term before removing
                self._archive_to_long_term(block)
                del self.short_term_memory[block.block_id]

    def _archive_to_long_term(self, block: MemoryBlock):
        """Archive block to long-term memory (PostgreSQL)"""
        if not self.postgres_conn:
            return

        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memory_blocks
                    (block_id, content, token_count, block_type, priority, tier,
                     created_at, accessed_at, access_count, summary, conversation_id, phase, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (block_id) DO UPDATE
                    SET accessed_at = EXCLUDED.accessed_at,
                        access_count = EXCLUDED.access_count,
                        tier = EXCLUDED.tier
                """, (
                    block.block_id, block.content, block.token_count, block.block_type,
                    block.priority, block.tier.value, block.created_at, block.accessed_at,
                    block.access_count, block.summary, block.conversation_id, block.phase,
                    Json(block.metadata)
                ))
                self.postgres_conn.commit()
        except Exception as e:
            print(f"Failed to archive to long-term memory: {e}")

    def retrieve_from_short_term(self, block_id: str) -> Optional[MemoryBlock]:
        """
        Retrieve full content from short-term memory

        This doesn't add it back to working memory - just provides access.
        """
        block = self.short_term_memory.get(block_id)
        if block:
            block.accessed_at = datetime.now()
            block.access_count += 1
            self.short_term_retrievals += 1
            return block
        return None

    def get_short_term_references(self, relevance_threshold: float = 0.0) -> List[MemoryReference]:
        """
        Get lightweight references to short-term memory

        These can be included in context as links without consuming many tokens.
        """
        references = []
        for block in self.short_term_memory.values():
            ref = MemoryReference(
                ref_id=block.block_id,
                summary=block.summary or block.content[:100],
                block_type=block.block_type,
                token_count=block.token_count,
                tier=MemoryTier.SHORT_TERM,
                created_at=block.created_at,
                relevance_score=1.0  # Could implement relevance scoring
            )
            if ref.relevance_score >= relevance_threshold:
                references.append(ref)

        return references

    def build_context_with_references(self, user_query: str, include_references: bool = True) -> str:
        """
        Build context with short-term memory references

        Creates a prompt that includes:
        1. Active working memory content
        2. Lightweight references to short-term memory
        3. User can ask to dereference specific references if needed
        """
        parts = []

        # Add working memory blocks (high priority first)
        working_blocks = sorted(self.working_memory, key=lambda b: b.priority, reverse=True)
        for block in working_blocks:
            parts.append(block.content)

        # Add short-term memory references if requested
        if include_references and self.short_term_memory:
            parts.append("\n## Recently Discussed (Short-Term Memory):")
            parts.append("You can reference these topics if relevant to the current query.\n")

            refs = self.get_short_term_references()
            for i, ref in enumerate(refs[:10], 1):  # Limit to 10 references
                parts.append(f"{i}. {ref.to_context_link()}")

            parts.append("\n(To recall full details, ask to dereference a specific memory reference)")

        # Add user query
        parts.append(f"\nUser Query: {user_query}")

        return "\n\n".join(parts)

    def dereference_memory(self, ref_id: str) -> Optional[str]:
        """
        Dereference a memory reference to get full content

        When AI or user wants to recall full details from a reference.
        """
        # Check short-term memory
        block = self.retrieve_from_short_term(ref_id)
        if block:
            return f"## Memory Reference {ref_id}:\n\n{block.content}"

        # Check long-term memory (PostgreSQL)
        if self.postgres_conn:
            try:
                with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT content FROM memory_blocks WHERE block_id = %s
                    """, (ref_id,))
                    result = cur.fetchone()
                    if result:
                        return f"## Memory Reference {ref_id}:\n\n{result['content']}"
            except Exception as e:
                print(f"Failed to retrieve from long-term memory: {e}")

        return None

    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            "working_memory": {
                "blocks": len(self.working_memory),
                "tokens": self.working_tokens,
                "max_tokens": self.max_working_tokens,
                "utilization": self.get_working_utilization(),
                "utilization_percent": f"{self.get_working_utilization():.1%}",
                "in_optimal_range": self.is_optimal_range(),
                "target_range": f"{self.target_min:.0%}-{self.target_max:.0%}"
            },
            "short_term_memory": {
                "blocks": len(self.short_term_memory),
                "max_blocks": self.short_term_max_blocks,
                "retrievals": self.short_term_retrievals
            },
            "long_term_memory": {
                "enabled": self.postgres_conn is not None,
                "blocks": self._count_long_term_blocks() if self.postgres_conn else 0
            },
            "compaction_count": self.compaction_count
        }

    def _count_long_term_blocks(self) -> int:
        """Count blocks in long-term memory"""
        if not self.postgres_conn:
            return 0

        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memory_blocks")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def close(self):
        """Close connections"""
        if self.postgres_conn:
            self.postgres_conn.close()


class DecayingMemoryManager:
    """
    Time-Based Decaying-Resolution Memory Manager
    Based on ai-that-works Episode #18: "Decaying-Resolution Memory"

    Implements time-based memory decay where older memories are automatically
    summarized to save tokens while maintaining temporal awareness.

    Decay Schedule:
    - < 1 hour: Full detail (resolution_level=0)
    - 1-24 hours: Summarized (resolution_level=1) - 50% token reduction
    - 24-168 hours (1 week): Compressed (resolution_level=2) - 70% token reduction
    - > 1 week: Archived (resolution_level=3) - moved to long-term only

    Benefits:
    - 30-50% token savings for long conversations
    - Extended context without token explosion
    - Temporal awareness maintained
    - Recent = detailed, Old = summarized
    """

    # Decay schedule configuration
    DECAY_SCHEDULE = {
        0: {
            "max_age_hours": 1,
            "resolution": 0,
            "label": "full detail",
            "token_reduction": 0.0
        },
        1: {
            "max_age_hours": 24,
            "resolution": 1,
            "label": "summarized",
            "token_reduction": 0.5  # 50% reduction
        },
        2: {
            "max_age_hours": 168,  # 1 week
            "resolution": 2,
            "label": "highly compressed",
            "token_reduction": 0.7  # 70% reduction
        },
        3: {
            "max_age_hours": float('inf'),
            "resolution": 3,
            "label": "archived",
            "token_reduction": 1.0  # Remove from working memory
        }
    }

    def __init__(self, hierarchical_memory: HierarchicalMemory, summarization_engine=None):
        """
        Initialize decaying memory manager

        Args:
            hierarchical_memory: HierarchicalMemory instance to manage
            summarization_engine: Optional AI engine for LLM-based summarization
        """
        self.memory = hierarchical_memory
        self.summarization_engine = summarization_engine
        self.tokens_saved = 0
        self.decay_operations = 0

    def calculate_target_resolution(self, block: MemoryBlock) -> int:
        """
        Calculate target resolution level based on block age

        Args:
            block: Memory block to evaluate

        Returns:
            Target resolution level (0-3)
        """
        age_hours = block.get_age_hours()

        for level in sorted(self.DECAY_SCHEDULE.keys()):
            if age_hours < self.DECAY_SCHEDULE[level]["max_age_hours"]:
                return level

        return 3  # Archived

    async def apply_decay(self, block: MemoryBlock) -> MemoryBlock:
        """
        Apply time-based decay to a single memory block

        Args:
            block: Memory block to decay

        Returns:
            Decayed memory block
        """
        target_resolution = self.calculate_target_resolution(block)

        # Already at target resolution
        if block.resolution_level >= target_resolution:
            return block

        # Save original content if not already saved
        if block.original_content is None:
            block.original_content = block.content
            block.original_tokens = block.token_count

        # Apply decay based on target resolution
        if target_resolution == 1:
            # Summarize (50% reduction)
            block.content = await self._summarize(block, reduction=0.5)
        elif target_resolution == 2:
            # Compress (70% reduction)
            block.content = await self._summarize(block, reduction=0.7)
        elif target_resolution == 3:
            # Archive (remove from working memory)
            block.tier = MemoryTier.LONG_TERM
            # Keep only minimal summary
            block.content = block.summary or block.content[:100] + "..."

        # Update block metadata
        new_tokens = self.memory._estimate_tokens(block.content)
        tokens_saved = block.token_count - new_tokens

        block.token_count = new_tokens
        block.resolution_level = target_resolution
        block.decay_applied = True

        # Track savings
        self.tokens_saved += tokens_saved
        self.decay_operations += 1

        return block

    async def _summarize(self, block: MemoryBlock, reduction: float) -> str:
        """
        Summarize block content using LLM or heuristic

        Args:
            block: Memory block to summarize
            reduction: Target token reduction (0.0-1.0)

        Returns:
            Summarized content
        """
        if self.summarization_engine:
            # Use LLM for intelligent summarization
            return await self._llm_summarize(block, reduction)
        else:
            # Fallback: heuristic summarization
            return self._heuristic_summarize(block, reduction)

    async def _llm_summarize(self, block: MemoryBlock, reduction: float) -> str:
        """
        Use LLM to intelligently summarize content

        Args:
            block: Memory block
            reduction: Target reduction ratio

        Returns:
            Summarized content
        """
        target_length = int(len(block.content) * (1 - reduction))

        prompt = f"""Summarize the following conversation memory concisely while preserving key information:

{block.content}

Reduce to approximately {target_length} characters while keeping the most important points."""

        try:
            # Try to use the engine's generate method
            if hasattr(self.summarization_engine, 'generate'):
                result = self.summarization_engine.generate(prompt, model_selection="fast")
                return result.get('response', block.content[:target_length])
            elif hasattr(self.summarization_engine, 'query'):
                result = self.summarization_engine.query(prompt, model="fast", use_rag=False, use_cache=False)
                return result.content
        except Exception as e:
            print(f"LLM summarization failed: {e}, using heuristic")
            return self._heuristic_summarize(block, reduction)

    def _heuristic_summarize(self, block: MemoryBlock, reduction: float) -> str:
        """
        Heuristic-based summarization (fallback)

        Strategy:
        - Keep first and last sentences
        - Remove middle content proportionally
        - Preserve code blocks and important markers

        Args:
            block: Memory block
            reduction: Target reduction ratio

        Returns:
            Summarized content
        """
        content = block.content
        target_length = int(len(content) * (1 - reduction))

        if len(content) <= target_length:
            return content

        # Split into sentences
        sentences = content.split('. ')

        if len(sentences) <= 3:
            # Too few sentences, just truncate
            return content[:target_length] + "..."

        # Keep first 2 and last 1 sentence
        summary_sentences = []
        summary_sentences.append(sentences[0])
        if len(sentences) > 1:
            summary_sentences.append(sentences[1])

        summary_sentences.append(f"[... {len(sentences) - 3} sentences omitted ...]")

        summary_sentences.append(sentences[-1])

        summary = '. '.join(summary_sentences)

        # If still too long, truncate
        if len(summary) > target_length:
            summary = summary[:target_length] + "..."

        return summary

    async def apply_decay_to_all(self, min_age_hours: float = 1.0) -> Dict[str, Any]:
        """
        Apply decay to all eligible blocks in working memory

        Args:
            min_age_hours: Minimum age before applying decay

        Returns:
            Statistics about decay operations
        """
        initial_tokens = self.memory.working_tokens
        decayed_blocks = []

        for block in self.memory.working_memory:
            age_hours = block.get_age_hours()

            if age_hours >= min_age_hours:
                target_resolution = self.calculate_target_resolution(block)

                if target_resolution > block.resolution_level:
                    decayed_block = await self.apply_decay(block)
                    decayed_blocks.append(decayed_block)

        # Recalculate total tokens
        self.memory.working_tokens = sum(b.token_count for b in self.memory.working_memory)
        final_tokens = self.memory.working_tokens
        tokens_saved = initial_tokens - final_tokens

        return {
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
            "tokens_saved": tokens_saved,
            "token_reduction_percent": (tokens_saved / initial_tokens * 100) if initial_tokens > 0 else 0,
            "blocks_decayed": len(decayed_blocks),
            "total_decay_operations": self.decay_operations,
            "total_tokens_saved": self.tokens_saved
        }

    def get_decay_schedule_info(self) -> List[Dict]:
        """Get information about decay schedule"""
        return [
            {
                "resolution_level": level,
                "max_age_hours": config["max_age_hours"],
                "label": config["label"],
                "token_reduction_percent": config["token_reduction"] * 100
            }
            for level, config in self.DECAY_SCHEDULE.items()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get decay statistics"""
        # Count blocks by resolution level
        resolution_counts = {}
        for block in self.memory.working_memory:
            level = block.resolution_level
            resolution_counts[level] = resolution_counts.get(level, 0) + 1

        return {
            "enabled": True,
            "tokens_saved_total": self.tokens_saved,
            "decay_operations": self.decay_operations,
            "blocks_by_resolution": resolution_counts,
            "decay_schedule": self.get_decay_schedule_info()
        }


# Example usage
if __name__ == "__main__":
    print("Hierarchical Memory System Test")
    print("=" * 60)

    # Initialize
    memory = HierarchicalMemory(max_working_tokens=8192)

    # Add content to working memory
    for i in range(10):
        memory.add_to_working_memory(
            f"This is conversation turn {i}. " + "Some content here. " * 50,
            block_type="user" if i % 2 == 0 else "assistant",
            priority=5 - (i // 2)  # Decreasing priority for older messages
        )

    print(f"\nWorking Memory: {memory.working_tokens} tokens")
    print(f"Utilization: {memory.get_working_utilization():.1%}")
    print(f"Blocks in working memory: {len(memory.working_memory)}")

    # Trigger compaction
    stats = memory.compact_to_short_term()
    print(f"\nCompaction Results:")
    print(f"  Freed {stats['tokens_freed']} tokens")
    print(f"  Compacted {stats['blocks_compacted']} blocks to short-term memory")
    print(f"  Working memory: {stats['final_tokens']} tokens ({stats['working_utilization']:.1%})")
    print(f"  Short-term memory: {stats['short_term_blocks']} blocks")

    # Get references
    refs = memory.get_short_term_references()
    print(f"\nShort-term memory references ({len(refs)}):")
    for ref in refs[:3]:
        print(f"  - {ref.to_context_link()}")

    # Build context with references
    context = memory.build_context_with_references("What did we discuss about turn 0?")
    print(f"\nContext with references ({len(context)} chars):")
    print(context[:500] + "...")

    # Dereference a memory
    if refs:
        dereferenced = memory.dereference_memory(refs[0].ref_id)
        print(f"\nDereferenced memory:")
        print(dereferenced[:200] + "...")

    # Stats
    stats = memory.get_stats()
    print(f"\nFinal Stats: {json.dumps(stats, indent=2)}")

    memory.close()

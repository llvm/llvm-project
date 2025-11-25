#!/usr/bin/env python3
"""
Enhanced Cognitive Memory System - Human Brain-Inspired Architecture

Improvements based on cognitive neuroscience research:

1. **Emotional Salience**: Memories tagged with importance/emotion persist longer
2. **Associative Networks**: Memories linked by semantic similarity and co-occurrence
3. **Consolidation**: Background process that strengthens important memories
4. **Context-Dependent Retrieval**: Memories surface based on current context
5. **Adaptive Decay**: Rarely accessed, low-importance memories gracefully fade
6. **Reconstructive Recall**: Build coherent responses from memory fragments
7. **Meta-Memory**: Track confidence and source of memories

Key Enhancements over hierarchical_memory.py:
- Importance scoring beyond simple priority
- Semantic clustering and association
- Emotional/salience tagging
- Consolidation process (like human sleep)
- Better retrieval based on relevance, not just recency
- Confidence tracking

Author: DSMIL Integration Framework
Version: 2.0.0
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import math

# Vector similarity for semantic associations
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Note: sentence-transformers not available. Using text-based associations.")

# PostgreSQL for persistence
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class MemoryTier(Enum):
    """Memory hierarchy tiers"""
    SENSORY = "sensory"          # Ultra-short term, sub-second
    WORKING = "working"          # Active context window
    SHORT_TERM = "short_term"    # Recently used, accessible
    LONG_TERM = "long_term"      # Consolidated, permanent
    ARCHIVED = "archived"        # Rarely accessed, compressed


class MemoryType(Enum):
    """Types of memory content"""
    EPISODIC = "episodic"        # Specific events/interactions
    SEMANTIC = "semantic"        # Facts and knowledge
    PROCEDURAL = "procedural"    # Skills and procedures
    WORKING = "working"          # Temporary task-related


class SalienceLevel(Enum):
    """Emotional/importance salience"""
    CRITICAL = 5      # Never forget
    HIGH = 4          # Very important
    MODERATE = 3      # Moderately important
    LOW = 2           # Can be forgotten
    TRIVIAL = 1       # Forget quickly


@dataclass
class CognitiveMemoryBlock:
    """Enhanced memory block with cognitive metadata"""
    block_id: str
    content: str
    token_count: int

    # Cognitive attributes
    memory_type: MemoryType = MemoryType.EPISODIC
    tier: MemoryTier = MemoryTier.WORKING
    salience: SalienceLevel = SalienceLevel.MODERATE

    # Temporal attributes
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_consolidated: Optional[datetime] = None
    access_count: int = 0

    # Quality attributes
    confidence: float = 1.0  # 0.0-1.0, tracks memory reliability
    source_quality: float = 1.0  # How reliable was the source

    # Association attributes
    semantic_embedding: Optional[np.ndarray] = None
    associated_blocks: Set[str] = field(default_factory=set)
    context_tags: Set[str] = field(default_factory=set)

    # Consolidation state
    consolidation_strength: float = 1.0  # How well consolidated (1.0-10.0)
    rehearsal_count: int = 0  # Number of times reinforced

    # Metadata
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    phase: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def calculate_importance_score(self) -> float:
        """
        Calculate composite importance score for retention decisions

        Factors:
        - Salience level (emotional/critical importance)
        - Access frequency (how often recalled)
        - Recency (when last accessed)
        - Consolidation strength (how well established)
        - Confidence (how reliable)
        """
        # Base salience (1-5)
        salience_score = self.salience.value

        # Access pattern (logarithmic to avoid over-weighting)
        access_score = math.log1p(self.access_count)

        # Recency (decay over time)
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        recency_score = math.exp(-hours_since_access / 24)  # Half-life of 24 hours

        # Consolidation bonus
        consolidation_score = self.consolidation_strength / 10.0

        # Confidence factor
        confidence_score = self.confidence

        # Weighted combination
        importance = (
            salience_score * 3.0 +
            access_score * 2.0 +
            recency_score * 1.5 +
            consolidation_score * 2.0 +
            confidence_score * 1.5
        )

        return importance

    def should_retain(self, threshold: float = 5.0) -> bool:
        """Determine if memory should be retained or allowed to decay"""
        return self.calculate_importance_score() >= threshold

    def access(self):
        """Record memory access (strengthens memory)"""
        self.last_accessed = datetime.now()
        self.access_count += 1

        # Spaced repetition strengthening
        if self.access_count % 3 == 0:
            self.consolidation_strength = min(10.0, self.consolidation_strength + 0.5)

    def to_dict(self) -> Dict:
        """Convert to dictionary (for serialization)"""
        data = asdict(self)
        data['tier'] = self.tier.value
        data['memory_type'] = self.memory_type.value
        data['salience'] = self.salience.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        if self.last_consolidated:
            data['last_consolidated'] = self.last_consolidated.isoformat()
        if self.semantic_embedding is not None:
            data['semantic_embedding'] = self.semantic_embedding.tolist()
        data['associated_blocks'] = list(self.associated_blocks)
        data['context_tags'] = list(self.context_tags)
        return data


class CognitiveMemorySystem:
    """
    Enhanced memory system based on human cognitive architecture

    Features:
    1. Multi-tier memory with adaptive transitions
    2. Importance-based retention (not just LRU)
    3. Semantic associations between memories
    4. Consolidation process (background strengthening)
    5. Context-dependent retrieval
    6. Confidence tracking and source quality
    7. Graceful forgetting of low-importance memories
    """

    def __init__(self,
                 max_working_tokens: int = 16384,
                 max_short_term_blocks: int = 100,
                 consolidation_interval_hours: float = 2.0,
                 enable_embeddings: bool = True,
                 postgres_config: Optional[Dict] = None):
        """Initialize enhanced cognitive memory system"""

        self.max_working_tokens = max_working_tokens
        self.max_short_term_blocks = max_short_term_blocks
        self.consolidation_interval_hours = consolidation_interval_hours

        # Memory stores by tier
        self.sensory_memory: List[CognitiveMemoryBlock] = []
        self.working_memory: List[CognitiveMemoryBlock] = []
        self.short_term_memory: Dict[str, CognitiveMemoryBlock] = {}
        self.long_term_memory: Dict[str, CognitiveMemoryBlock] = {}

        # Working memory token tracking
        self.working_tokens = 0

        # Semantic embedding model
        self.embedder = None
        if enable_embeddings and EMBEDDINGS_AVAILABLE:
            print("Loading semantic embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Semantic associations enabled")

        # Association graph (block_id -> set of related block_ids)
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)

        # Context tags index (tag -> set of block_ids)
        self.context_index: Dict[str, Set[str]] = defaultdict(set)

        # Long-term storage (PostgreSQL)
        self.postgres_conn = None
        if POSTGRES_AVAILABLE and postgres_config:
            try:
                self.postgres_conn = psycopg2.connect(**postgres_config)
                self._init_schema()
            except Exception as e:
                print(f"PostgreSQL connection failed: {e}")

        # Consolidation state
        self.last_consolidation = datetime.now()

        # Statistics
        self.stats = {
            'total_memories': 0,
            'consolidations': 0,
            'forgotten_memories': 0,
            'associations_created': 0
        }

    def _init_schema(self):
        """Initialize PostgreSQL schema for long-term memory"""
        if not self.postgres_conn:
            return

        with self.postgres_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_memories (
                    block_id VARCHAR(32) PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR(20),
                    tier VARCHAR(20),
                    salience INTEGER,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    last_consolidated TIMESTAMP,
                    access_count INTEGER,
                    confidence FLOAT,
                    source_quality FLOAT,
                    consolidation_strength FLOAT,
                    rehearsal_count INTEGER,
                    semantic_embedding FLOAT[],
                    associated_blocks TEXT[],
                    context_tags TEXT[],
                    conversation_id VARCHAR(64),
                    user_id VARCHAR(64),
                    metadata JSONB
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_accessed
                ON cognitive_memories(last_accessed DESC)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_salience
                ON cognitive_memories(salience DESC)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_context
                ON cognitive_memories USING GIN(context_tags)
            """)

            self.postgres_conn.commit()

    def _generate_block_id(self, content: str) -> str:
        """Generate unique block ID"""
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return int(len(text) * 0.25)

    def _compute_semantic_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute semantic embedding for association"""
        if self.embedder is None:
            return None
        try:
            return self.embedder.encode(text, convert_to_numpy=True)
        except:
            return None

    def _extract_context_tags(self, text: str, metadata: Dict) -> Set[str]:
        """Extract context tags for indexing"""
        tags = set()

        # From metadata
        if 'tags' in metadata:
            tags.update(metadata['tags'])
        if 'topic' in metadata:
            tags.add(metadata['topic'].lower())
        if 'category' in metadata:
            tags.add(metadata['category'].lower())

        # Simple keyword extraction (could use NLP for better results)
        # Look for capitalized words (likely entities/topics)
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 3:
                tags.add(word.lower())

        return tags

    def add_memory(self,
                   content: str,
                   memory_type: MemoryType = MemoryType.EPISODIC,
                   salience: SalienceLevel = SalienceLevel.MODERATE,
                   confidence: float = 1.0,
                   source_quality: float = 1.0,
                   conversation_id: Optional[str] = None,
                   user_id: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> CognitiveMemoryBlock:
        """
        Add new memory to the system

        The memory enters working memory and may transition through tiers
        based on importance and access patterns.
        """
        metadata = metadata or {}

        block = CognitiveMemoryBlock(
            block_id=self._generate_block_id(content),
            content=content,
            token_count=self._estimate_tokens(content),
            memory_type=memory_type,
            tier=MemoryTier.WORKING,
            salience=salience,
            confidence=confidence,
            source_quality=source_quality,
            conversation_id=conversation_id,
            user_id=user_id,
            metadata=metadata
        )

        # Compute semantic embedding
        block.semantic_embedding = self._compute_semantic_embedding(content)

        # Extract context tags
        block.context_tags = self._extract_context_tags(content, metadata)
        for tag in block.context_tags:
            self.context_index[tag].add(block.block_id)

        # Find semantic associations
        self._create_associations(block)

        # Add to working memory
        self.working_memory.append(block)
        self.working_tokens += block.token_count
        self.stats['total_memories'] += 1

        # Trigger management if needed
        self._manage_memory_capacity()

        return block

    def _create_associations(self, new_block: CognitiveMemoryBlock):
        """Create semantic associations with existing memories"""
        if new_block.semantic_embedding is None:
            return

        # Check similarity with recent memories
        candidates = []
        candidates.extend(self.working_memory[-20:])  # Recent working memory
        candidates.extend(list(self.short_term_memory.values())[-50:])  # Recent short-term

        for candidate in candidates:
            if candidate.block_id == new_block.block_id:
                continue
            if candidate.semantic_embedding is None:
                continue

            # Compute cosine similarity
            similarity = np.dot(new_block.semantic_embedding, candidate.semantic_embedding) / (
                np.linalg.norm(new_block.semantic_embedding) * np.linalg.norm(candidate.semantic_embedding)
            )

            # Create association if similar enough
            if similarity > 0.7:  # Threshold for association
                new_block.associated_blocks.add(candidate.block_id)
                candidate.associated_blocks.add(new_block.block_id)
                self.association_graph[new_block.block_id].add(candidate.block_id)
                self.association_graph[candidate.block_id].add(new_block.block_id)
                self.stats['associations_created'] += 1

    def _manage_memory_capacity(self):
        """Manage memory capacity across tiers"""
        # Working memory management
        if self.working_tokens > self.max_working_tokens * 0.8:
            self._transition_to_short_term()

        # Short-term memory management
        if len(self.short_term_memory) > self.max_short_term_blocks:
            self._consolidate_or_forget()

        # Periodic consolidation
        hours_since_consolidation = (datetime.now() - self.last_consolidation).total_seconds() / 3600
        if hours_since_consolidation >= self.consolidation_interval_hours:
            self._background_consolidation()

    def _transition_to_short_term(self):
        """Move memories from working to short-term based on importance"""
        # Calculate importance scores
        scored = [(block, block.calculate_importance_score()) for block in self.working_memory]
        scored.sort(key=lambda x: x[1])  # Lowest importance first

        # Move lower-importance memories to short-term
        target_tokens = int(self.max_working_tokens * 0.5)
        freed_tokens = 0

        while self.working_tokens - freed_tokens > target_tokens and scored:
            block, score = scored.pop(0)

            block.tier = MemoryTier.SHORT_TERM
            self.short_term_memory[block.block_id] = block
            freed_tokens += block.token_count

        # Update working memory
        self.working_memory = [b for b, s in scored]
        self.working_tokens -= freed_tokens

    def _consolidate_or_forget(self):
        """Consolidate important memories to long-term or forget unimportant ones"""
        for block_id, block in list(self.short_term_memory.items()):
            importance = block.calculate_importance_score()

            # High importance: consolidate to long-term
            if importance >= 8.0:
                block.tier = MemoryTier.LONG_TERM
                block.last_consolidated = datetime.now()
                block.consolidation_strength = min(10.0, block.consolidation_strength + 1.0)
                self.long_term_memory[block_id] = block
                self._persist_to_postgres(block)
                del self.short_term_memory[block_id]

            # Very low importance: forget
            elif importance < 2.0:
                self._forget_memory(block)
                del self.short_term_memory[block_id]

    def _background_consolidation(self):
        """
        Background consolidation process (like human sleep)

        Strengthens important memories, weakens unimportant ones
        """
        print("🧠 Background consolidation starting...")

        # Consolidate across all tiers
        all_memories = []
        all_memories.extend(self.working_memory)
        all_memories.extend(self.short_term_memory.values())
        all_memories.extend(self.long_term_memory.values())

        for block in all_memories:
            importance = block.calculate_importance_score()

            # Strengthen important memories
            if importance >= 7.0:
                block.consolidation_strength = min(10.0, block.consolidation_strength + 0.5)
                block.rehearsal_count += 1

            # Weaken unimportant memories
            elif importance < 3.0:
                block.consolidation_strength = max(0.1, block.consolidation_strength - 0.3)
                block.confidence = max(0.1, block.confidence - 0.1)

        self.last_consolidation = datetime.now()
        self.stats['consolidations'] += 1

        print(f"✓ Consolidation complete. Memories processed: {len(all_memories)}")

    def _forget_memory(self, block: CognitiveMemoryBlock):
        """Gracefully forget a memory (remove associations)"""
        # Remove from association graph
        for associated_id in block.associated_blocks:
            if associated_id in self.association_graph:
                self.association_graph[associated_id].discard(block.block_id)
        del self.association_graph[block.block_id]

        # Remove from context index
        for tag in block.context_tags:
            self.context_index[tag].discard(block.block_id)

        self.stats['forgotten_memories'] += 1

    def _persist_to_postgres(self, block: CognitiveMemoryBlock):
        """Persist consolidated memory to PostgreSQL"""
        if not self.postgres_conn:
            return

        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO cognitive_memories VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (block_id) DO UPDATE
                    SET last_accessed = EXCLUDED.last_accessed,
                        access_count = EXCLUDED.access_count,
                        consolidation_strength = EXCLUDED.consolidation_strength,
                        tier = EXCLUDED.tier
                """, (
                    block.block_id, block.content, block.memory_type.value, block.tier.value,
                    block.salience.value, block.created_at, block.last_accessed,
                    block.last_consolidated, block.access_count, block.confidence,
                    block.source_quality, block.consolidation_strength, block.rehearsal_count,
                    block.semantic_embedding.tolist() if block.semantic_embedding is not None else None,
                    list(block.associated_blocks), list(block.context_tags),
                    block.conversation_id, block.user_id, Json(block.metadata)
                ))
                self.postgres_conn.commit()
        except Exception as e:
            print(f"Failed to persist memory: {e}")

    def retrieve_by_context(self, context: str, max_results: int = 10) -> List[CognitiveMemoryBlock]:
        """
        Retrieve memories based on current context

        Uses semantic similarity and context tags
        """
        # Compute query embedding
        query_embedding = self._compute_semantic_embedding(context)

        # Extract context tags
        query_tags = self._extract_context_tags(context, {})

        # Collect candidates
        candidates = []
        candidates.extend(self.working_memory)
        candidates.extend(self.short_term_memory.values())
        candidates.extend(self.long_term_memory.values())

        # Score each candidate
        scored = []
        for block in candidates:
            score = 0.0

            # Semantic similarity (if available)
            if query_embedding is not None and block.semantic_embedding is not None:
                similarity = np.dot(query_embedding, block.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(block.semantic_embedding)
                )
                score += similarity * 5.0

            # Context tag overlap
            tag_overlap = len(query_tags & block.context_tags)
            score += tag_overlap * 2.0

            # Importance bonus
            importance = block.calculate_importance_score()
            score += importance * 0.5

            # Confidence factor
            score *= block.confidence

            if score > 0:
                scored.append((block, score))

        # Sort by score and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [block for block, score in scored[:max_results]]

        # Update access patterns
        for block in results:
            block.access()

        return results

    def get_memory(self, block_id: str) -> Optional[CognitiveMemoryBlock]:
        """Get memory by block ID"""
        # Search in all tiers
        for block in self.working_memory:
            if block.block_id == block_id:
                block.access()
                return block

        if block_id in self.short_term_memory:
            block = self.short_term_memory[block_id]
            block.access()
            return block

        if block_id in self.long_term_memory:
            block = self.long_term_memory[block_id]
            block.access()
            return block

        return None

    def retrieve_associated(self, block_id: str, max_depth: int = 2) -> List[CognitiveMemoryBlock]:
        """Retrieve memories associated with given block (spreading activation)"""
        visited = set()
        results = []
        queue = [(block_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Get memory block
            block = None
            if current_id in self.short_term_memory:
                block = self.short_term_memory[current_id]
            elif current_id in self.long_term_memory:
                block = self.long_term_memory[current_id]
            else:
                for wm_block in self.working_memory:
                    if wm_block.block_id == current_id:
                        block = wm_block
                        break

            if block:
                results.append(block)
                block.access()

                # Add associated blocks to queue
                for assoc_id in block.associated_blocks:
                    if assoc_id not in visited:
                        queue.append((assoc_id, depth + 1))

        return results[1:]  # Exclude the original block

    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            'memory_counts': {
                'sensory': len(self.sensory_memory),
                'working': len(self.working_memory),
                'short_term': len(self.short_term_memory),
                'long_term': len(self.long_term_memory),
                'total': self.stats['total_memories']
            },
            'working_memory': {
                'tokens': self.working_tokens,
                'max_tokens': self.max_working_tokens,
                'utilization': self.working_tokens / self.max_working_tokens
            },
            'associations': {
                'graph_size': len(self.association_graph),
                'total_created': self.stats['associations_created'],
                'context_tags': len(self.context_index)
            },
            'consolidation': {
                'count': self.stats['consolidations'],
                'last': self.last_consolidation.isoformat(),
                'hours_since': (datetime.now() - self.last_consolidation).total_seconds() / 3600
            },
            'forgotten_memories': self.stats['forgotten_memories']
        }

    def close(self):
        """Clean shutdown"""
        if self.postgres_conn:
            self.postgres_conn.close()


# Example usage
if __name__ == "__main__":
    print("Enhanced Cognitive Memory System Test")
    print("=" * 60)

    # Initialize
    memory = CognitiveMemorySystem(
        max_working_tokens=8192,
        enable_embeddings=True
    )

    # Add memories with different salience levels
    memory.add_memory(
        "The user's name is John and he prefers Python over JavaScript.",
        memory_type=MemoryType.SEMANTIC,
        salience=SalienceLevel.HIGH,
        metadata={'topic': 'user_preferences'}
    )

    memory.add_memory(
        "We discussed implementing a new feature for the dashboard.",
        memory_type=MemoryType.EPISODIC,
        salience=SalienceLevel.MODERATE
    )

    memory.add_memory(
        "The weather is nice today.",
        memory_type=MemoryType.EPISODIC,
        salience=SalienceLevel.TRIVIAL
    )

    # Retrieve by context
    print("\n🔍 Retrieving memories related to 'user preferences':")
    results = memory.retrieve_by_context("What are the user's preferences?")
    for block in results:
        print(f"  - {block.content[:80]}... (importance: {block.calculate_importance_score():.1f})")

    # Get stats
    stats = memory.get_stats()
    print(f"\n📊 Memory Statistics:")
    print(json.dumps(stats, indent=2))

    memory.close()

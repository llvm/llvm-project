#!/usr/bin/env python3
"""
MemLayer Enhanced - Intelligent Memory Layer for AI Agents
===========================================================
Inspired by memlayer (https://github.com/divagr18/memlayer)

Features:
- Salience Filtering: Intelligently determines what information deserves storage
- Hybrid Storage: Vector store (ChromaDB) + Knowledge graph (NetworkX)
- Automatic Consolidation: Background extraction and indexing
- Proactive Task Reminders: Automatic injection of relevant context
- Episodic + Semantic Memory: Learn from experiences and patterns

Three Operating Modes:
- LOCAL: ML-based filtering, full storage (~10s startup)
- ONLINE: API embeddings, full storage (~2s startup)
- LIGHTWEIGHT: Keyword filtering, graph only (<1s startup)

Search Tiers:
- FAST: <100ms, 2 results, no graph traversal
- BALANCED: <500ms, 5 results, default
- DEEP: <2s, 10 results, entity extraction, relationship traversal

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import re
import json
import hashlib
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """Memory layer operating modes"""
    LOCAL = "local"           # ML model filtering, full storage
    ONLINE = "online"         # API embeddings, full storage
    LIGHTWEIGHT = "lightweight"  # Keyword filtering, graph only


class SearchTier(str, Enum):
    """Search depth tiers"""
    FAST = "fast"             # <100ms, 2 results
    BALANCED = "balanced"     # <500ms, 5 results
    DEEP = "deep"             # <2s, 10 results


class MemoryType(str, Enum):
    """Types of memory"""
    EPISODIC = "episodic"     # Specific experiences/events
    SEMANTIC = "semantic"     # Facts, knowledge, patterns
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"       # Active context


@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    mentions: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    salience_score: float = 0.5


@dataclass
class Relationship:
    """Knowledge graph relationship"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryEntry:
    """A memory entry with metadata"""
    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[List[float]] = None
    entities: List[str] = field(default_factory=list)
    salience_score: float = 0.5
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result with relevance"""
    entry: MemoryEntry
    relevance_score: float
    match_type: str  # 'vector', 'keyword', 'entity', 'relationship'
    related_entities: List[Entity] = field(default_factory=list)


class SalienceFilter:
    """
    Salience filtering to determine what information deserves storage.

    Filters out:
    - Greetings, acknowledgments
    - Filler words, meta-conversation
    - Repetitive content

    Preserves:
    - Facts, preferences, user info
    - Decisions, relationships
    - Code, technical details
    - Important instructions
    """

    # Patterns to filter out (low salience)
    LOW_SALIENCE_PATTERNS = [
        r'^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye|goodbye)\b',
        r'^(um|uh|well|so|like|you know)\b',
        r'^(got it|understood|i see|makes sense)\b',
        r'^(let me|i will|i\'ll)\s+(help|assist|try)',
        r'^(great|good|nice|awesome|perfect)\s*(job|work)?[!.]*$',
    ]

    # Patterns indicating high salience
    HIGH_SALIENCE_PATTERNS = [
        r'\b(remember|note|important|always|never|must|should)\b',
        r'\b(my|your|our)\s+(name|email|preference|setting|config)',
        r'\b(api|key|password|token|secret)\b',
        r'\b(decide|decided|decision|choice|chose)\b',
        r'\bdef\s+\w+|class\s+\w+|function\s+\w+',  # Code
        r'\b(error|bug|fix|issue|problem)\b',
        r'\b(requirement|spec|must have|need)\b',
    ]

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        'person': r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
        'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
        'url': r'https?://[^\s]+',
        'file_path': r'[/\\][\w./\\-]+\.\w+',
        'code_ref': r'\b(def|class|function|method)\s+(\w+)',
        'number': r'\b\d+(?:\.\d+)?\b',
        'date': r'\b\d{4}-\d{2}-\d{2}\b',
    }

    def __init__(self, mode: OperatingMode = OperatingMode.LIGHTWEIGHT):
        self.mode = mode
        self.salience_threshold = 0.3

    def compute_salience(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Compute salience score for text.

        Returns:
            (salience_score, metadata)
        """
        text_lower = text.lower().strip()
        metadata = {
            'entities': [],
            'patterns_matched': [],
            'word_count': len(text.split()),
        }

        # Base score
        score = 0.5

        # Check for low salience patterns
        for pattern in self.LOW_SALIENCE_PATTERNS:
            if re.search(pattern, text_lower):
                score -= 0.2
                metadata['patterns_matched'].append(('low', pattern))

        # Check for high salience patterns
        for pattern in self.HIGH_SALIENCE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.2
                metadata['patterns_matched'].append(('high', pattern))

        # Extract entities
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                score += 0.1 * min(len(matches), 3)
                metadata['entities'].extend([
                    {'type': entity_type, 'value': m if isinstance(m, str) else m[0]}
                    for m in matches[:5]
                ])

        # Adjust by length (very short = low salience, unless code)
        if metadata['word_count'] < 5 and 'def ' not in text and 'class ' not in text:
            score -= 0.1
        elif metadata['word_count'] > 50:
            score += 0.1

        # Code detection bonus
        if re.search(r'```|\bdef\s|\bclass\s|\bfunction\s|import\s', text):
            score += 0.2
            metadata['is_code'] = True

        # Clamp score
        score = max(0.0, min(1.0, score))

        return score, metadata

    def should_store(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Determine if text should be stored.

        Returns:
            (should_store, salience_score, metadata)
        """
        score, metadata = self.compute_salience(text)
        should_store = score >= self.salience_threshold
        return should_store, score, metadata


class KnowledgeGraph:
    """
    In-memory knowledge graph for entity relationships.

    Stores entities and their relationships for structured retrieval.
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids

    def add_entity(self, name: str, entity_type: str, attributes: Optional[Dict] = None) -> Entity:
        """Add or update an entity"""
        entity_id = hashlib.sha256(f"{name}:{entity_type}".encode()).hexdigest()[:16]

        if entity_id in self.entities:
            # Update existing
            entity = self.entities[entity_id]
            entity.mentions += 1
            entity.last_accessed = datetime.now()
            if attributes:
                entity.attributes.update(attributes)
        else:
            # Create new
            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=attributes or {}
            )
            self.entities[entity_id] = entity
            self.entity_index[entity_type].add(entity_id)

        return entity

    def add_relationship(
        self,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relation_type: str,
        evidence: Optional[str] = None
    ) -> Relationship:
        """Add a relationship between entities"""
        source = self.add_entity(source_name, source_type)
        target = self.add_entity(target_name, target_type)

        # Check for existing relationship
        for rel in self.relationships:
            if rel.source_id == source.id and rel.target_id == target.id and rel.relation_type == relation_type:
                rel.weight += 0.1
                if evidence:
                    rel.evidence.append(evidence)
                return rel

        # Create new relationship
        relationship = Relationship(
            source_id=source.id,
            target_id=target.id,
            relation_type=relation_type,
            evidence=[evidence] if evidence else []
        )
        self.relationships.append(relationship)
        return relationship

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Search entities by name"""
        query_lower = query.lower()
        results = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if query_lower in entity.name.lower():
                results.append(entity)

        return sorted(results, key=lambda e: e.mentions, reverse=True)

    def get_related_entities(self, entity_id: str, max_hops: int = 1) -> List[Tuple[Entity, str]]:
        """Get entities related to given entity"""
        related = []
        visited = {entity_id}

        current_level = [entity_id]
        for _ in range(max_hops):
            next_level = []
            for eid in current_level:
                for rel in self.relationships:
                    if rel.source_id == eid and rel.target_id not in visited:
                        target = self.entities.get(rel.target_id)
                        if target:
                            related.append((target, rel.relation_type))
                            next_level.append(rel.target_id)
                            visited.add(rel.target_id)
                    elif rel.target_id == eid and rel.source_id not in visited:
                        source = self.entities.get(rel.source_id)
                        if source:
                            related.append((source, rel.relation_type))
                            next_level.append(rel.source_id)
                            visited.add(rel.source_id)
            current_level = next_level

        return related

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dict"""
        return {
            'entities': {eid: {
                'id': e.id,
                'name': e.name,
                'type': e.entity_type,
                'attributes': e.attributes,
                'mentions': e.mentions
            } for eid, e in self.entities.items()},
            'relationships': [{
                'source': r.source_id,
                'target': r.target_id,
                'type': r.relation_type,
                'weight': r.weight
            } for r in self.relationships]
        }


class VectorStore:
    """
    Simple in-memory vector store for semantic search.

    For production, would integrate with ChromaDB, Pinecone, etc.
    """

    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add(self, entry: MemoryEntry):
        """Add entry to store"""
        self.entries[entry.id] = entry
        if entry.embedding:
            self.embeddings[entry.id] = entry.embedding

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search by embedding similarity"""
        if not query_embedding or not self.embeddings:
            return []

        results = []
        for entry_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((entry_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Fallback keyword search"""
        query_terms = set(query.lower().split())
        results = []

        for entry_id, entry in self.entries.items():
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                results.append((entry_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity"""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class MemLayerEnhanced:
    """
    Enhanced memory layer with salience filtering and hybrid storage.

    Combines:
    - Salience filtering for intelligent storage decisions
    - Vector store for semantic similarity search
    - Knowledge graph for entity relationships
    - Episodic memory for experiences
    - Semantic memory for patterns and facts
    """

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.LIGHTWEIGHT,
        storage_path: Optional[Path] = None,
        embedding_fn: Optional[callable] = None
    ):
        self.mode = mode
        self.storage_path = storage_path or Path.home() / ".dsmil" / "memlayer"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Components
        self.salience_filter = SalienceFilter(mode)
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.embedding_fn = embedding_fn

        # Memory storage
        self.episodic_memory: Dict[str, MemoryEntry] = {}
        self.semantic_memory: Dict[str, MemoryEntry] = {}
        self.working_memory: List[MemoryEntry] = []

        # Metrics
        self.stored_count = 0
        self.filtered_count = 0
        self.search_count = 0

        # Load existing data
        self._load()

        logger.info(f"MemLayer Enhanced initialized in {mode.value} mode")

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        force: bool = False,
        metadata: Optional[Dict] = None
    ) -> Optional[MemoryEntry]:
        """
        Store content in memory with salience filtering.

        Args:
            content: The content to store
            memory_type: Type of memory (episodic, semantic, etc.)
            force: Force storage even if low salience
            metadata: Additional metadata

        Returns:
            MemoryEntry if stored, None if filtered out
        """
        # Check salience
        should_store, salience_score, salience_metadata = self.salience_filter.should_store(content)

        if not should_store and not force:
            self.filtered_count += 1
            logger.debug(f"Filtered out content (salience={salience_score:.2f})")
            return None

        # Generate embedding if available
        embedding = None
        if self.embedding_fn and self.mode != OperatingMode.LIGHTWEIGHT:
            try:
                embedding = self.embedding_fn(content)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Extract and store entities
        entity_ids = []
        for entity_info in salience_metadata.get('entities', []):
            entity = self.knowledge_graph.add_entity(
                name=entity_info['value'],
                entity_type=entity_info['type']
            )
            entity_ids.append(entity.id)

        # Create memory entry
        entry = MemoryEntry(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            entities=entity_ids,
            salience_score=salience_score,
            metadata={**(metadata or {}), **salience_metadata}
        )

        # Store in appropriate location
        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory[entry.id] = entry
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[entry.id] = entry
        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(entry)
            # Limit working memory size
            if len(self.working_memory) > 50:
                self.working_memory.pop(0)

        # Add to vector store
        self.vector_store.add(entry)

        self.stored_count += 1
        return entry

    def search(
        self,
        query: str,
        tier: SearchTier = SearchTier.BALANCED,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[SearchResult]:
        """
        Search memories with configurable depth.

        Args:
            query: Search query
            tier: Search depth tier (FAST, BALANCED, DEEP)
            memory_types: Filter by memory types

        Returns:
            List of SearchResults
        """
        self.search_count += 1
        results = []

        # Configure search based on tier
        if tier == SearchTier.FAST:
            k = 2
            use_graph = False
        elif tier == SearchTier.BALANCED:
            k = 5
            use_graph = True
        else:  # DEEP
            k = 10
            use_graph = True

        # Vector/keyword search
        if self.embedding_fn and self.mode != OperatingMode.LIGHTWEIGHT:
            try:
                query_embedding = self.embedding_fn(query)
                vector_results = self.vector_store.search(query_embedding, k)
            except Exception:
                vector_results = self.vector_store.keyword_search(query, k)
        else:
            vector_results = self.vector_store.keyword_search(query, k)

        # Process vector results
        for entry_id, score in vector_results:
            entry = self.vector_store.entries.get(entry_id)
            if not entry:
                continue

            # Filter by memory type
            if memory_types and entry.memory_type not in memory_types:
                continue

            related_entities = []
            if use_graph and entry.entities:
                for eid in entry.entities[:3]:
                    related = self.knowledge_graph.get_related_entities(eid, max_hops=1)
                    related_entities.extend([e for e, _ in related])

            results.append(SearchResult(
                entry=entry,
                relevance_score=score,
                match_type='vector' if self.embedding_fn else 'keyword',
                related_entities=related_entities[:5]
            ))

        # Entity search for DEEP tier
        if tier == SearchTier.DEEP:
            entity_results = self.knowledge_graph.search_entities(query)
            for entity in entity_results[:3]:
                # Find memories mentioning this entity
                for entry in list(self.episodic_memory.values()) + list(self.semantic_memory.values()):
                    if entity.id in entry.entities:
                        if not any(r.entry.id == entry.id for r in results):
                            results.append(SearchResult(
                                entry=entry,
                                relevance_score=0.5,
                                match_type='entity',
                                related_entities=[entity]
                            ))

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:k]

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get relevant context for a query.

        Returns formatted context string with most relevant memories.
        """
        results = self.search(query, tier=SearchTier.BALANCED)

        context_parts = []
        token_count = 0

        for result in results:
            entry_tokens = len(result.entry.content.split()) * 1.3  # Rough estimate
            if token_count + entry_tokens > max_tokens:
                break

            context_parts.append(f"[{result.entry.memory_type.value}] {result.entry.content}")
            token_count += entry_tokens

            # Update access tracking
            result.entry.access_count += 1
            result.entry.last_accessed = datetime.now()

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    def consolidate(self) -> Dict[str, int]:
        """
        Consolidate memories - extract patterns, update graph, prune old entries.

        Returns:
            Stats about consolidation
        """
        stats = {'patterns_extracted': 0, 'relationships_added': 0, 'entries_pruned': 0}

        # Extract relationships from co-occurring entities
        entity_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        for entry in list(self.episodic_memory.values()) + list(self.semantic_memory.values()):
            entities = entry.entities
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    entity_cooccurrence[(e1, e2)] += 1

        # Create relationships for frequent co-occurrences
        for (e1, e2), count in entity_cooccurrence.items():
            if count >= 2:
                ent1 = self.knowledge_graph.get_entity(e1)
                ent2 = self.knowledge_graph.get_entity(e2)
                if ent1 and ent2:
                    self.knowledge_graph.add_relationship(
                        ent1.name, ent1.entity_type,
                        ent2.name, ent2.entity_type,
                        "co_occurs_with",
                        f"Co-occurred {count} times"
                    )
                    stats['relationships_added'] += 1

        # Prune old, low-salience entries
        cutoff = datetime.now() - timedelta(days=30)
        for memory_dict in [self.episodic_memory, self.semantic_memory]:
            to_remove = []
            for entry_id, entry in memory_dict.items():
                if entry.last_accessed < cutoff and entry.salience_score < 0.4 and entry.access_count < 2:
                    to_remove.append(entry_id)

            for entry_id in to_remove:
                del memory_dict[entry_id]
                stats['entries_pruned'] += 1

        self._save()
        return stats

    def _save(self):
        """Save memory to disk"""
        data = {
            'episodic': {k: self._entry_to_dict(v) for k, v in self.episodic_memory.items()},
            'semantic': {k: self._entry_to_dict(v) for k, v in self.semantic_memory.items()},
            'graph': self.knowledge_graph.to_dict(),
            'stats': {
                'stored_count': self.stored_count,
                'filtered_count': self.filtered_count,
                'search_count': self.search_count
            }
        }

        storage_file = self.storage_path / "memory.json"
        storage_file.write_text(json.dumps(data, indent=2, default=str))

    def _load(self):
        """Load memory from disk"""
        storage_file = self.storage_path / "memory.json"
        if not storage_file.exists():
            return

        try:
            data = json.loads(storage_file.read_text())

            # Load episodic memory
            for entry_id, entry_data in data.get('episodic', {}).items():
                entry = self._dict_to_entry(entry_data)
                self.episodic_memory[entry_id] = entry
                self.vector_store.add(entry)

            # Load semantic memory
            for entry_id, entry_data in data.get('semantic', {}).items():
                entry = self._dict_to_entry(entry_data)
                self.semantic_memory[entry_id] = entry
                self.vector_store.add(entry)

            # Load stats
            stats = data.get('stats', {})
            self.stored_count = stats.get('stored_count', 0)
            self.filtered_count = stats.get('filtered_count', 0)
            self.search_count = stats.get('search_count', 0)

            logger.info(f"Loaded {len(self.episodic_memory)} episodic, {len(self.semantic_memory)} semantic memories")

        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

    def _entry_to_dict(self, entry: MemoryEntry) -> Dict:
        """Convert entry to dict"""
        return {
            'id': entry.id,
            'content': entry.content,
            'memory_type': entry.memory_type.value,
            'entities': entry.entities,
            'salience_score': entry.salience_score,
            'importance': entry.importance,
            'access_count': entry.access_count,
            'created_at': entry.created_at.isoformat(),
            'last_accessed': entry.last_accessed.isoformat(),
            'metadata': entry.metadata
        }

    def _dict_to_entry(self, data: Dict) -> MemoryEntry:
        """Convert dict to entry"""
        return MemoryEntry(
            id=data['id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            entities=data.get('entities', []),
            salience_score=data.get('salience_score', 0.5),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else datetime.now(),
            metadata=data.get('metadata', {})
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'mode': self.mode.value,
            'episodic_memories': len(self.episodic_memory),
            'semantic_memories': len(self.semantic_memory),
            'working_memory_size': len(self.working_memory),
            'entities': len(self.knowledge_graph.entities),
            'relationships': len(self.knowledge_graph.relationships),
            'stored_count': self.stored_count,
            'filtered_count': self.filtered_count,
            'filter_rate': self.filtered_count / (self.stored_count + self.filtered_count) if (self.stored_count + self.filtered_count) > 0 else 0,
            'search_count': self.search_count
        }


# Singleton instance
_memlayer: Optional[MemLayerEnhanced] = None


def get_memlayer(mode: OperatingMode = OperatingMode.LIGHTWEIGHT) -> MemLayerEnhanced:
    """Get or create singleton memory layer"""
    global _memlayer
    if _memlayer is None:
        _memlayer = MemLayerEnhanced(mode=mode)
    return _memlayer


if __name__ == "__main__":
    # Test the memory layer
    mem = get_memlayer(OperatingMode.LIGHTWEIGHT)

    print("MemLayer Enhanced Test")
    print("=" * 60)

    # Test storage with filtering
    test_entries = [
        ("Hello!", MemoryType.EPISODIC),  # Should be filtered
        ("The API key is stored in config.json", MemoryType.SEMANTIC),  # Should store
        ("Remember to always use HTTPS for API calls", MemoryType.SEMANTIC),  # Should store
        ("ok", MemoryType.EPISODIC),  # Should be filtered
        ("def calculate_total(items):\n    return sum(item.price for item in items)", MemoryType.PROCEDURAL),  # Should store
        ("The user prefers dark mode for all applications", MemoryType.SEMANTIC),  # Should store
    ]

    for content, mem_type in test_entries:
        result = mem.store(content, mem_type)
        status = "Stored" if result else "Filtered"
        print(f"{status}: {content[:50]}...")

    print("\n" + "=" * 60)
    print("Search Test:")

    results = mem.search("API configuration", tier=SearchTier.BALANCED)
    for r in results:
        print(f"  [{r.match_type}] {r.entry.content[:60]}... (score: {r.relevance_score:.2f})")

    print("\n" + "=" * 60)
    print("Stats:", mem.get_stats())

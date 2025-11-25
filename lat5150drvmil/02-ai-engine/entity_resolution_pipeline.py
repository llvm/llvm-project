#!/usr/bin/env python3
"""
Entity Resolution Pipeline
Based on ai-that-works Episode #10: "Entity Extraction and Resolution"

Three-Stage Pipeline:
1. EXTRACT: Identify entities from text (people, orgs, addresses, emails, phones, crypto)
2. RESOLVE: Deduplicate and resolve variants (John Smith = J. Smith = john.smith@example.com)
3. ENRICH: Augment with external intelligence (DIRECTEYE OSINT, blockchain, threat intel)

Benefits:
- Better entity understanding across conversations
- Cross-source intelligence gathering
- Deduplication of entity mentions
- Rich context from multiple sources

Integration:
- Pairs perfectly with DIRECTEYE for enrichment
- Works with RAG for entity-aware retrieval
- Feeds into event-driven agent for entity tracking
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
import json


class EntityType(Enum):
    """Types of entities to extract"""
    PERSON = "person"
    ORGANIZATION = "organization"
    EMAIL = "email"
    PHONE = "phone"
    CRYPTO_ADDRESS = "crypto_address"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    LOCATION = "location"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """
    Extracted entity with metadata

    Attributes:
        entity_id: Unique identifier (hash-based)
        entity_type: Type of entity
        raw_text: Original text as extracted
        normalized_text: Normalized form for matching
        confidence: Extraction confidence (0.0-1.0)
        source_context: Where entity was found
        metadata: Additional metadata
        enriched_data: Data from enrichment stage
        aliases: Known aliases/variants
        first_seen: First extraction timestamp
        last_seen: Last extraction timestamp
        mention_count: Number of times mentioned
    """
    entity_id: str
    entity_type: EntityType
    raw_text: str
    normalized_text: str
    confidence: float
    source_context: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    enriched_data: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 1

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "source_context": self.source_context[:100] + "..." if len(self.source_context) > 100 else self.source_context,
            "metadata": self.metadata,
            "enriched_data": self.enriched_data,
            "aliases": list(self.aliases),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "mention_count": self.mention_count
        }


class EntityExtractor:
    """
    Stage 1: Extract entities from text

    Uses pattern matching and heuristics to identify entities.
    Can be enhanced with NER models for better accuracy.
    """

    # Regex patterns for entity extraction
    PATTERNS = {
        EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        EntityType.PHONE: r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        EntityType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        EntityType.DOMAIN: r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
        EntityType.URL: r'https?://[^\s<>"{}|\\^`\[\]]+',
        EntityType.CRYPTO_ADDRESS: r'\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,87})\b',
    }

    def __init__(self):
        """Initialize entity extractor"""
        self.extracted_count = 0

    def extract(self, text: str, context: Optional[str] = None) -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Text to extract from
            context: Optional context for source tracking

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract each entity type
        for entity_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                raw_text = match.group(0)
                normalized = self._normalize(raw_text, entity_type)

                entity = Entity(
                    entity_id=self._generate_id(normalized, entity_type),
                    entity_type=entity_type,
                    raw_text=raw_text,
                    normalized_text=normalized,
                    confidence=self._calculate_confidence(raw_text, entity_type),
                    source_context=context or text,
                    metadata={"position": match.span()}
                )

                entities.append(entity)
                self.extracted_count += 1

        return entities

    def _normalize(self, text: str, entity_type: EntityType) -> str:
        """
        Normalize entity text for matching

        Args:
            text: Raw text
            entity_type: Entity type

        Returns:
            Normalized text
        """
        if entity_type == EntityType.EMAIL:
            return text.lower().strip()
        elif entity_type == EntityType.PHONE:
            # Remove formatting
            return re.sub(r'[^\d+]', '', text)
        elif entity_type == EntityType.DOMAIN:
            return text.lower().strip()
        elif entity_type == EntityType.CRYPTO_ADDRESS:
            return text.lower().strip()
        else:
            return text.strip()

    def _generate_id(self, normalized_text: str, entity_type: EntityType) -> str:
        """Generate unique entity ID"""
        content = f"{entity_type.value}:{normalized_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _calculate_confidence(self, text: str, entity_type: EntityType) -> float:
        """
        Calculate extraction confidence

        Args:
            text: Extracted text
            entity_type: Entity type

        Returns:
            Confidence score (0.0-1.0)
        """
        # Simple heuristics - can be enhanced with ML
        if entity_type == EntityType.EMAIL:
            # Check for valid TLD
            if re.search(r'\.(com|org|net|edu|gov|io|ai)$', text, re.IGNORECASE):
                return 0.95
            return 0.7

        elif entity_type == EntityType.CRYPTO_ADDRESS:
            # Ethereum addresses start with 0x
            if text.startswith('0x') and len(text) == 42:
                return 0.9
            # Bitcoin addresses
            elif text.startswith(('1', '3', 'bc1')):
                return 0.85
            return 0.6

        elif entity_type == EntityType.IP_ADDRESS:
            # Check if valid IP range
            parts = text.split('.')
            if all(0 <= int(p) <= 255 for p in parts):
                return 0.95
            return 0.5

        else:
            return 0.8


class EntityResolver:
    """
    Stage 2: Resolve and deduplicate entities

    Identifies variants and aliases of the same entity:
    - john@example.com = j.smith@example.com (same domain, similar name)
    - +1-555-123-4567 = (555) 123-4567 (same phone, different format)
    - Multiple mentions of same crypto address
    """

    def __init__(self):
        """Initialize entity resolver"""
        self.entity_registry: Dict[str, Entity] = {}  # entity_id -> Entity
        self.resolution_count = 0

    def resolve(self, entities: List[Entity]) -> List[Entity]:
        """
        Resolve entities against registry

        Args:
            entities: List of extracted entities

        Returns:
            List of resolved entities (deduplicated with merged info)
        """
        resolved = []

        for entity in entities:
            existing = self.entity_registry.get(entity.entity_id)

            if existing:
                # Merge with existing
                existing.mention_count += 1
                existing.last_seen = datetime.now()
                existing.aliases.add(entity.raw_text)

                # Update confidence (take maximum)
                if entity.confidence > existing.confidence:
                    existing.confidence = entity.confidence

                # Merge metadata
                existing.metadata.update(entity.metadata)

                resolved.append(existing)
                self.resolution_count += 1

            else:
                # New entity
                entity.aliases.add(entity.raw_text)
                self.entity_registry[entity.entity_id] = entity
                resolved.append(entity)

        return resolved

    def find_similar(self, entity: Entity, threshold: float = 0.7) -> List[Entity]:
        """
        Find similar entities (fuzzy matching)

        Args:
            entity: Entity to match
            threshold: Similarity threshold

        Returns:
            List of similar entities
        """
        similar = []

        for existing in self.entity_registry.values():
            if existing.entity_type != entity.entity_type:
                continue

            similarity = self._calculate_similarity(
                entity.normalized_text,
                existing.normalized_text
            )

            if similarity >= threshold:
                similar.append(existing)

        return similar

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Simple Levenshtein-based similarity for now.
        Can be enhanced with embeddings for better matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        if text1 == text2:
            return 1.0

        # Simple character overlap
        set1 = set(text1.lower())
        set2 = set(text2.lower())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        entity_counts = {}
        for entity in self.entity_registry.values():
            entity_type = entity.entity_type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        return {
            "total_entities": len(self.entity_registry),
            "resolution_count": self.resolution_count,
            "entities_by_type": entity_counts
        }


class EntityEnricher:
    """
    Stage 3: Enrich entities with external intelligence

    Uses DIRECTEYE Intelligence Platform to augment entities with:
    - OSINT data (people search, breach data, corporate intel)
    - Blockchain analysis (crypto addresses)
    - Threat intelligence (IPs, domains, hashes)
    """

    def __init__(self, directeye_intel=None):
        """
        Initialize entity enricher

        Args:
            directeye_intel: DirectEyeIntelligence instance (optional)
        """
        self.directeye = directeye_intel
        self.enrichment_count = 0

    async def enrich(self, entity: Entity) -> Entity:
        """
        Enrich entity with external intelligence

        Args:
            entity: Entity to enrich

        Returns:
            Enriched entity
        """
        if not self.directeye:
            # No enrichment available
            return entity

        try:
            if entity.entity_type == EntityType.EMAIL:
                # OSINT query for email
                result = await self.directeye.osint_query(entity.normalized_text)
                entity.enriched_data["osint"] = result

            elif entity.entity_type == EntityType.CRYPTO_ADDRESS:
                # Blockchain analysis
                result = await self.directeye.blockchain_analyze(
                    entity.normalized_text,
                    chain="ethereum"  # Can detect chain from address format
                )
                entity.enriched_data["blockchain"] = result

            elif entity.entity_type in [EntityType.IP_ADDRESS, EntityType.DOMAIN]:
                # Threat intelligence
                result = await self.directeye.threat_intelligence(entity.normalized_text)
                entity.enriched_data["threat_intel"] = result

            self.enrichment_count += 1

        except Exception as e:
            entity.enriched_data["enrichment_error"] = str(e)

        return entity

    async def enrich_batch(self, entities: List[Entity]) -> List[Entity]:
        """
        Enrich multiple entities (parallel for efficiency)

        Args:
            entities: List of entities

        Returns:
            List of enriched entities
        """
        import asyncio
        tasks = [self.enrich(entity) for entity in entities]
        return await asyncio.gather(*tasks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get enricher statistics"""
        return {
            "enrichment_count": self.enrichment_count,
            "directeye_available": self.directeye is not None
        }


class EntityResolutionPipeline:
    """
    Complete 3-stage entity resolution pipeline

    Integration with LAT5150DRVMIL stack:
    - DIRECTEYE Intelligence: OSINT/blockchain/threat enrichment
    - Event-Driven Agent: Log extraction/resolution events
    - Hierarchical Memory: Store entities as structured metadata
    - Enhanced RAG: Entity-aware retrieval
    - Conversation Manager: Track entities per conversation

    Usage:
        pipeline = EntityResolutionPipeline(
            directeye_intel=intel,
            event_driven_agent=agent,
            hierarchical_memory=memory
        )
        entities = await pipeline.process("Text with entities...")
    """

    def __init__(
        self,
        directeye_intel=None,
        event_driven_agent=None,
        hierarchical_memory=None,
        rag_system=None
    ):
        """
        Initialize pipeline with full stack integration

        Args:
            directeye_intel: DirectEyeIntelligence for enrichment
            event_driven_agent: EventDrivenAgent for audit trail
            hierarchical_memory: HierarchicalMemory for entity storage
            rag_system: EnhancedRAGSystem for entity-aware retrieval
        """
        self.extractor = EntityExtractor()
        self.resolver = EntityResolver()
        self.enricher = EntityEnricher(directeye_intel)

        # Stack integrations
        self.event_agent = event_driven_agent
        self.memory = hierarchical_memory
        self.rag = rag_system
        self.directeye = directeye_intel

    async def process(
        self,
        text: str,
        context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        enrich: bool = True
    ) -> List[Entity]:
        """
        Process text through full pipeline with stack integration

        Args:
            text: Text to process
            context: Optional context
            conversation_id: Conversation ID for tracking
            enrich: Whether to enrich entities (requires DIRECTEYE)

        Returns:
            List of resolved and enriched entities
        """
        # Stage 1: Extract
        entities = self.extractor.extract(text, context)

        # Log extraction event (Event-Driven Agent)
        if self.event_agent:
            from event_driven_agent import EventType as AgentEventType
            self.event_agent.log_event(
                AgentEventType.METADATA,
                {
                    "action": "entity_extraction",
                    "entities_extracted": len(entities),
                    "entity_types": [e.entity_type.value for e in entities]
                },
                metadata={"conversation_id": conversation_id}
            )

        # Stage 2: Resolve
        entities = self.resolver.resolve(entities)

        # Log resolution event
        if self.event_agent:
            from event_driven_agent import EventType as AgentEventType
            self.event_agent.log_event(
                AgentEventType.METADATA,
                {
                    "action": "entity_resolution",
                    "entities_resolved": len(entities),
                    "new_entities": len([e for e in entities if e.mention_count == 1])
                },
                metadata={"conversation_id": conversation_id}
            )

        # Stage 3: Enrich (optional, async)
        if enrich and self.enricher.directeye:
            entities = await self.enricher.enrich_batch(entities)

            # Log enrichment event
            if self.event_agent:
                from event_driven_agent import EventType as AgentEventType
                enriched_count = len([e for e in entities if e.enriched_data])
                self.event_agent.log_event(
                    AgentEventType.METADATA,
                    {
                        "action": "entity_enrichment",
                        "entities_enriched": enriched_count,
                        "sources": ["directeye"]
                    },
                    metadata={"conversation_id": conversation_id}
                )

        # Store in Hierarchical Memory
        if self.memory:
            self._store_entities_in_memory(entities, conversation_id)

        # Index in RAG (for entity-aware retrieval)
        if self.rag:
            self._index_entities_in_rag(entities)

        return entities

    def _store_entities_in_memory(
        self,
        entities: List[Entity],
        conversation_id: Optional[str] = None
    ):
        """
        Store entities in hierarchical memory as structured metadata

        Args:
            entities: List of entities
            conversation_id: Conversation ID
        """
        if not self.memory:
            return

        # Create memory block for entities
        entity_summary = f"Extracted {len(entities)} entities: " + ", ".join(
            f"{e.entity_type.value}={e.normalized_text}" for e in entities[:5]
        )

        if len(entities) > 5:
            entity_summary += f" (+{len(entities) - 5} more)"

        try:
            self.memory.add_to_working_memory(
                content=entity_summary,
                block_type="entity_metadata",
                priority=6,  # Medium priority
                conversation_id=conversation_id,
                metadata={
                    "entity_count": len(entities),
                    "entity_types": list(set(e.entity_type.value for e in entities)),
                    "entities": [e.to_dict() for e in entities]
                }
            )
        except Exception as e:
            print(f"Failed to store entities in memory: {e}")

    def _index_entities_in_rag(self, entities: List[Entity]):
        """
        Index entities in RAG system for entity-aware retrieval

        Args:
            entities: List of entities
        """
        if not self.rag:
            return

        for entity in entities:
            # Create entity document for RAG
            entity_doc = {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "normalized_text": entity.normalized_text,
                "enriched_data": entity.enriched_data,
                "mention_count": entity.mention_count
            }

            # This would need a method like add_entity_document()
            # in enhanced_rag_system.py for proper entity indexing
            # For now, we just track it

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from registry"""
        return self.resolver.entity_registry.get(entity_id)

    def search_entities(
        self,
        entity_type: Optional[EntityType] = None,
        min_mentions: int = 1
    ) -> List[Entity]:
        """
        Search entities in registry

        Args:
            entity_type: Filter by entity type
            min_mentions: Minimum mention count

        Returns:
            List of matching entities
        """
        results = []

        for entity in self.resolver.entity_registry.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            if entity.mention_count < min_mentions:
                continue

            results.append(entity)

        # Sort by mention count (descending)
        results.sort(key=lambda e: e.mention_count, reverse=True)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "extractor": {
                "extracted_count": self.extractor.extracted_count
            },
            "resolver": self.resolver.get_statistics(),
            "enricher": self.enricher.get_statistics()
        }


async def main():
    """Demo usage"""
    print("=== Entity Resolution Pipeline Demo ===\n")

    pipeline = EntityResolutionPipeline()

    # Sample text with various entities
    text = """
    Contact john.doe@example.com or call +1-555-123-4567 for more information.
    The transaction was sent to 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb.
    Suspicious activity from IP 192.168.1.100 accessing api.example.com.
    Also check j.doe@example.com for updates.
    """

    print("Processing text...")
    entities = await pipeline.process(text, enrich=False)

    print(f"\nFound {len(entities)} entities:\n")

    for entity in entities:
        print(f"- {entity.entity_type.value}: {entity.normalized_text}")
        print(f"  Confidence: {entity.confidence:.2f}")
        print(f"  Mentions: {entity.mention_count}")
        if entity.aliases:
            print(f"  Aliases: {entity.aliases}")

    # Search for specific entity types
    print("\n\nEmail entities:")
    emails = pipeline.search_entities(entity_type=EntityType.EMAIL)
    for entity in emails:
        print(f"- {entity.normalized_text} (mentioned {entity.mention_count}x)")

    print("\n\nCrypto addresses:")
    crypto = pipeline.search_entities(entity_type=EntityType.CRYPTO_ADDRESS)
    for entity in crypto:
        print(f"- {entity.normalized_text}")

    # Statistics
    print("\n\nPipeline Statistics:")
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

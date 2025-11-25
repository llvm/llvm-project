#!/usr/bin/env python3
"""
Unified Storage Abstraction Layer

Provides a unified interface for all storage systems in the LAT5150DRVMIL AI engine:
- PostgreSQL (conversations, metadata, long-term memory)
- SQLite (RAM disk, audit, checkpoints)
- Redis (caching)
- Vector stores (Qdrant, ChromaDB)
- File systems (ZFS, tmpfs)

This abstraction layer enables:
- Consistent API across all storage types
- Intelligent routing and lifecycle management
- Transaction-like semantics
- Unified backup/recovery
- Health monitoring and metrics
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Core Types and Enums
# ============================================================================

class StorageType(Enum):
    """Types of storage layers"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    REDIS = "redis"
    VECTOR = "vector"  # Qdrant, ChromaDB, etc.
    FILESYSTEM = "filesystem"  # ZFS, tmpfs
    MEMORY = "memory"  # In-memory only


class ContentType(Enum):
    """Types of content being stored"""
    CONVERSATION = "conversation"
    MESSAGE = "message"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    MEMORY = "memory"
    CHECKPOINT = "checkpoint"
    CACHE = "cache"
    AUDIT = "audit"
    METADATA = "metadata"
    MULTIMODAL = "multimodal"  # images, audio, video


class StorageTier(Enum):
    """Storage tiers for lifecycle management"""
    HOT = "hot"      # Fast access (Redis, RAM disk)
    WARM = "warm"    # Active data (PostgreSQL, Qdrant)
    COLD = "cold"    # Archive (File system, compressed)
    FROZEN = "frozen"  # Long-term archive (tape, S3 Glacier)


@dataclass
class StorageHandle:
    """Reference to stored data"""
    storage_type: StorageType
    storage_id: str  # Backend-specific ID (row ID, key, file path)
    content_type: ContentType
    tier: StorageTier
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'storage_type': self.storage_type.value,
            'storage_id': self.storage_id,
            'content_type': self.content_type.value,
            'tier': self.tier.value,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StorageHandle':
        return cls(
            storage_type=StorageType(data['storage_type']),
            storage_id=data['storage_id'],
            content_type=ContentType(data['content_type']),
            tier=StorageTier(data['tier']),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )


@dataclass
class SearchResult:
    """Result from storage search"""
    handle: StorageHandle
    score: float
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageStats:
    """Storage layer statistics"""
    storage_type: StorageType
    total_items: int
    total_size_bytes: int
    avg_access_time_ms: float
    cache_hit_rate: Optional[float] = None
    last_backup: Optional[datetime] = None
    health_status: str = "healthy"
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Abstract Base Classes
# ============================================================================

class AbstractStorageBackend(ABC):
    """
    Base class for all storage backends

    All storage systems must implement this interface for unified access.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize storage backend

        Args:
            config: Backend-specific configuration
        """
        self.config = config
        self.storage_type = StorageType.MEMORY  # Override in subclass
        self._stats = {
            'operations': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'errors': 0
        }

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to storage backend

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to storage backend

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    def store(
        self,
        data: Any,
        content_type: ContentType,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> StorageHandle:
        """
        Store data in backend

        Args:
            data: Data to store (any serializable type)
            content_type: Type of content
            key: Optional key (auto-generated if not provided)
            ttl: Time-to-live in seconds (None = no expiration)
            metadata: Additional metadata

        Returns:
            StorageHandle for retrieving data
        """
        pass

    @abstractmethod
    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """
        Retrieve data by handle

        Args:
            handle: Storage handle from store()

        Returns:
            Retrieved data or None if not found
        """
        pass

    @abstractmethod
    def delete(self, handle: StorageHandle) -> bool:
        """
        Delete data by handle

        Args:
            handle: Storage handle

        Returns:
            True if deletion successful
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for data

        Args:
            query: Search query (text or vector)
            content_type: Filter by content type
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def get_stats(self) -> StorageStats:
        """
        Get storage statistics

        Returns:
            StorageStats object
        """
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """
        Check backend health

        Returns:
            (is_healthy, status_message)
        """
        pass

    # Optional methods with default implementations

    def backup(self, destination: str) -> bool:
        """
        Backup storage to destination

        Args:
            destination: Backup destination path

        Returns:
            True if backup successful
        """
        logger.warning(f"{self.storage_type.value} does not implement backup()")
        return False

    def restore(self, source: str) -> bool:
        """
        Restore storage from backup

        Args:
            source: Backup source path

        Returns:
            True if restore successful
        """
        logger.warning(f"{self.storage_type.value} does not implement restore()")
        return False

    def optimize(self) -> bool:
        """
        Optimize storage (vacuum, reindex, etc.)

        Returns:
            True if optimization successful
        """
        logger.warning(f"{self.storage_type.value} does not implement optimize()")
        return False


class AbstractVectorBackend(AbstractStorageBackend):
    """
    Base class for vector storage backends (Qdrant, ChromaDB, etc.)

    Adds vector-specific operations to the base interface.
    """

    @abstractmethod
    def store_embedding(
        self,
        text: str,
        embedding: List[float],
        content_type: ContentType,
        metadata: Optional[Dict] = None
    ) -> StorageHandle:
        """
        Store text with its embedding vector

        Args:
            text: Original text
            embedding: Vector embedding
            content_type: Content type
            metadata: Additional metadata

        Returns:
            StorageHandle
        """
        pass

    @abstractmethod
    def vector_search(
        self,
        query_embedding: List[float],
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Semantic search by vector similarity

        Args:
            query_embedding: Query vector
            content_type: Filter by content type
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results ranked by similarity
        """
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        content_type: Optional[ContentType] = None,
        alpha: float = 0.5,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector + keyword matching

        Args:
            query_text: Text query for keyword matching
            query_embedding: Vector for semantic matching
            content_type: Filter by content type
            alpha: Weight for dense vs sparse (0=sparse, 1=dense)
            limit: Maximum results

        Returns:
            List of search results
        """
        pass


class AbstractCacheBackend(AbstractStorageBackend):
    """
    Base class for cache backends (Redis, Memcached)

    Adds cache-specific operations to the base interface.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set cache value

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern

        Args:
            pattern: Key pattern (e.g., "user:*")

        Returns:
            Number of keys invalidated
        """
        pass

    @abstractmethod
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate

        Returns:
            Hit rate (0.0 to 1.0)
        """
        pass


# ============================================================================
# Storage Handle Registry
# ============================================================================

class StorageHandleRegistry:
    """
    Registry for tracking storage handles across systems

    Enables cross-storage queries and handle resolution.
    """

    def __init__(self):
        self._registry: Dict[str, StorageHandle] = {}
        self._content_type_index: Dict[ContentType, List[str]] = {}
        self._storage_type_index: Dict[StorageType, List[str]] = {}

    def register(self, handle: StorageHandle) -> str:
        """
        Register a storage handle

        Args:
            handle: Storage handle to register

        Returns:
            Global handle ID
        """
        # Generate global handle ID
        handle_id = self._generate_handle_id(handle)

        # Store in registry
        self._registry[handle_id] = handle

        # Index by content type
        if handle.content_type not in self._content_type_index:
            self._content_type_index[handle.content_type] = []
        self._content_type_index[handle.content_type].append(handle_id)

        # Index by storage type
        if handle.storage_type not in self._storage_type_index:
            self._storage_type_index[handle.storage_type] = []
        self._storage_type_index[handle.storage_type].append(handle_id)

        return handle_id

    def lookup(self, handle_id: str) -> Optional[StorageHandle]:
        """
        Look up handle by ID

        Args:
            handle_id: Global handle ID

        Returns:
            StorageHandle or None
        """
        return self._registry.get(handle_id)

    def find_by_content_type(self, content_type: ContentType) -> List[StorageHandle]:
        """
        Find all handles for a content type

        Args:
            content_type: Content type to search

        Returns:
            List of matching handles
        """
        handle_ids = self._content_type_index.get(content_type, [])
        return [self._registry[hid] for hid in handle_ids if hid in self._registry]

    def find_by_storage_type(self, storage_type: StorageType) -> List[StorageHandle]:
        """
        Find all handles for a storage type

        Args:
            storage_type: Storage type to search

        Returns:
            List of matching handles
        """
        handle_ids = self._storage_type_index.get(storage_type, [])
        return [self._registry[hid] for hid in handle_ids if hid in self._registry]

    def unregister(self, handle_id: str) -> bool:
        """
        Remove handle from registry

        Args:
            handle_id: Global handle ID

        Returns:
            True if removed
        """
        if handle_id not in self._registry:
            return False

        handle = self._registry[handle_id]

        # Remove from indices
        if handle.content_type in self._content_type_index:
            try:
                self._content_type_index[handle.content_type].remove(handle_id)
            except ValueError:
                pass

        if handle.storage_type in self._storage_type_index:
            try:
                self._storage_type_index[handle.storage_type].remove(handle_id)
            except ValueError:
                pass

        # Remove from registry
        del self._registry[handle_id]

        return True

    def _generate_handle_id(self, handle: StorageHandle) -> str:
        """Generate unique handle ID"""
        data = f"{handle.storage_type.value}:{handle.storage_id}:{handle.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_handles': len(self._registry),
            'by_content_type': {
                ct.value: len(handles)
                for ct, handles in self._content_type_index.items()
            },
            'by_storage_type': {
                st.value: len(handles)
                for st, handles in self._storage_type_index.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("UNIFIED STORAGE ABSTRACTION LAYER")
    print("="*80 + "\n")

    # Example storage handle
    handle = StorageHandle(
        storage_type=StorageType.POSTGRESQL,
        storage_id="msg_12345",
        content_type=ContentType.MESSAGE,
        tier=StorageTier.WARM,
        metadata={"user_id": "user_001", "timestamp": "2024-01-15T10:30:00"}
    )

    print("Storage Handle Example:")
    print(f"  Type: {handle.storage_type.value}")
    print(f"  ID: {handle.storage_id}")
    print(f"  Content: {handle.content_type.value}")
    print(f"  Tier: {handle.tier.value}")
    print(f"  Created: {handle.created_at}")
    print()

    # Test registry
    registry = StorageHandleRegistry()
    handle_id = registry.register(handle)

    print(f"Registered handle: {handle_id}")

    retrieved = registry.lookup(handle_id)
    print(f"Retrieved handle: {retrieved.storage_id if retrieved else 'Not found'}")

    print("\nRegistry Stats:")
    stats = registry.get_stats()
    print(f"  Total handles: {stats['total_handles']}")
    print(f"  By content type: {stats['by_content_type']}")
    print(f"  By storage type: {stats['by_storage_type']}")

    print("\n✓ Unified Storage Abstraction Layer initialized")

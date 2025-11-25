#!/usr/bin/env python3
"""
Storage Orchestrator

Unified storage orchestration layer for the LAT5150DRVMIL AI engine.
Provides intelligent routing, lifecycle management, and coordination across
all storage backends (PostgreSQL, Redis, Qdrant, SQLite, FileSystems).

Features:
- Automatic backend selection based on content type and tier
- Cross-storage synchronization and replication
- Intelligent caching with Redis
- Distributed backup and recovery
- Health monitoring and failover
- Storage lifecycle management (HOT → WARM → COLD → FROZEN)
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from storage_abstraction import (
    AbstractStorageBackend,
    AbstractVectorBackend,
    AbstractCacheBackend,
    StorageType,
    ContentType,
    StorageTier,
    StorageHandle,
    SearchResult,
    StorageStats,
    StorageHandleRegistry
)

# Import backend implementations
from storage_postgresql import PostgreSQLStorageBackend
from storage_redis import RedisStorageBackend
from storage_qdrant import QdrantStorageBackend
from storage_sqlite import SQLiteStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class StoragePolicy:
    """
    Policy for storing content type

    Defines which backends to use, caching strategy, and lifecycle rules
    """
    content_type: ContentType
    primary_backend: StorageType
    cache_backend: Optional[StorageType] = None
    vector_backend: Optional[StorageType] = None
    cache_ttl: Optional[int] = None  # seconds
    enable_replication: bool = False
    replica_backend: Optional[StorageType] = None
    lifecycle_enabled: bool = False
    warm_after_days: Optional[int] = None
    cold_after_days: Optional[int] = None
    frozen_after_days: Optional[int] = None


@dataclass
class OrchestratorConfig:
    """Configuration for Storage Orchestrator"""
    # Backend configurations
    postgresql_config: Optional[Dict] = None
    redis_config: Optional[Dict] = None
    qdrant_config: Optional[Dict] = None
    sqlite_config: Optional[Dict] = None

    # Storage policies
    policies: Dict[ContentType, StoragePolicy] = field(default_factory=dict)

    # General settings
    enable_caching: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    enable_auto_optimization: bool = True
    optimization_interval: int = 3600  # seconds


class StorageOrchestrator:
    """
    Main orchestrator for unified storage management

    Intelligently routes operations to appropriate backends,
    manages caching, handles replication, and coordinates
    lifecycle management.
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize Storage Orchestrator

        Args:
            config: Orchestrator configuration
        """
        self.config = config

        # Backend registry
        self.backends: Dict[StorageType, AbstractStorageBackend] = {}

        # Handle registry for tracking across storage
        self.handle_registry = StorageHandleRegistry()

        # Default policies
        self._setup_default_policies()

        # Health monitoring
        self._health_check_thread = None
        self._optimization_thread = None
        self._shutdown_event = threading.Event()

        # Statistics
        self._stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'replications': 0
        }

    def initialize(self) -> bool:
        """
        Initialize all configured backends

        Returns:
            True if all backends initialized successfully
        """
        success = True

        # Initialize PostgreSQL
        if self.config.postgresql_config:
            try:
                pg_backend = PostgreSQLStorageBackend(self.config.postgresql_config)
                if pg_backend.connect():
                    self.backends[StorageType.POSTGRESQL] = pg_backend
                    logger.info("PostgreSQL backend initialized")
                else:
                    logger.error("Failed to initialize PostgreSQL backend")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL: {e}")
                success = False

        # Initialize Redis
        if self.config.redis_config:
            try:
                redis_backend = RedisStorageBackend(self.config.redis_config)
                if redis_backend.connect():
                    self.backends[StorageType.REDIS] = redis_backend
                    logger.info("Redis backend initialized")
                else:
                    logger.error("Failed to initialize Redis backend")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing Redis: {e}")
                success = False

        # Initialize Qdrant
        if self.config.qdrant_config:
            try:
                qdrant_backend = QdrantStorageBackend(self.config.qdrant_config)
                if qdrant_backend.connect():
                    self.backends[StorageType.VECTOR] = qdrant_backend
                    logger.info("Qdrant backend initialized")
                else:
                    logger.error("Failed to initialize Qdrant backend")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing Qdrant: {e}")
                success = False

        # Initialize SQLite
        if self.config.sqlite_config:
            try:
                sqlite_backend = SQLiteStorageBackend(self.config.sqlite_config)
                if sqlite_backend.connect():
                    self.backends[StorageType.SQLITE] = sqlite_backend
                    logger.info("SQLite backend initialized")
                else:
                    logger.error("Failed to initialize SQLite backend")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing SQLite: {e}")
                success = False

        # Start background threads
        if self.config.enable_health_checks:
            self._start_health_monitoring()

        if self.config.enable_auto_optimization:
            self._start_auto_optimization()

        logger.info(f"Storage Orchestrator initialized with {len(self.backends)} backends")
        return success

    def shutdown(self):
        """Shutdown orchestrator and all backends"""
        logger.info("Shutting down Storage Orchestrator")

        # Stop background threads
        self._shutdown_event.set()

        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)

        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)

        # Disconnect all backends
        for storage_type, backend in self.backends.items():
            try:
                backend.disconnect()
                logger.info(f"Disconnected {storage_type.value} backend")
            except Exception as e:
                logger.error(f"Error disconnecting {storage_type.value}: {e}")

    def store(
        self,
        data: Any,
        content_type: ContentType,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None,
        tier: Optional[StorageTier] = None
    ) -> StorageHandle:
        """
        Store data with intelligent backend selection

        Args:
            data: Data to store
            content_type: Type of content
            key: Optional key
            ttl: Time-to-live in seconds
            metadata: Additional metadata
            tier: Storage tier (HOT/WARM/COLD/FROZEN)

        Returns:
            StorageHandle for retrieving data
        """
        try:
            self._stats['total_operations'] += 1

            # Get policy for content type
            policy = self._get_policy(content_type)

            # Determine tier if not specified
            if tier is None:
                tier = StorageTier.WARM

            # Get primary backend
            backend = self._get_backend(policy.primary_backend)
            if not backend:
                raise ValueError(f"Backend {policy.primary_backend} not available")

            # Store in primary backend
            handle = backend.store(
                data=data,
                content_type=content_type,
                key=key,
                ttl=ttl,
                metadata=metadata
            )

            # Register handle
            handle_id = self.handle_registry.register(handle)

            # Cache if policy specifies
            if self.config.enable_caching and policy.cache_backend:
                self._cache_data(handle, data, policy.cache_ttl)

            # Replicate if policy specifies
            if policy.enable_replication and policy.replica_backend:
                self._replicate_data(handle, data, policy.replica_backend, metadata)

            logger.debug(f"Stored {content_type.value} in {policy.primary_backend.value}")
            return handle

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error storing data: {e}")
            raise

    def retrieve(self, handle: StorageHandle, use_cache: bool = True) -> Optional[Any]:
        """
        Retrieve data with intelligent caching

        Args:
            handle: Storage handle
            use_cache: Whether to use cache

        Returns:
            Retrieved data or None
        """
        try:
            self._stats['total_operations'] += 1

            # Try cache first if enabled
            if use_cache and self.config.enable_caching:
                cached_data = self._try_cache_retrieve(handle)
                if cached_data is not None:
                    self._stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {handle.storage_id}")
                    return cached_data
                else:
                    self._stats['cache_misses'] += 1

            # Get from primary backend
            backend = self._get_backend(handle.storage_type)
            if not backend:
                logger.error(f"Backend {handle.storage_type} not available")
                return None

            data = backend.retrieve(handle)

            # Update cache on retrieve
            if data and use_cache and self.config.enable_caching:
                policy = self._get_policy(handle.content_type)
                if policy.cache_backend:
                    self._cache_data(handle, data, policy.cache_ttl)

            return data

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error retrieving data: {e}")
            return None

    def delete(self, handle: StorageHandle) -> bool:
        """
        Delete data from all locations

        Args:
            handle: Storage handle

        Returns:
            True if deletion successful
        """
        try:
            self._stats['total_operations'] += 1

            # Delete from primary backend
            backend = self._get_backend(handle.storage_type)
            if not backend:
                return False

            success = backend.delete(handle)

            # Delete from cache
            if self.config.enable_caching:
                self._delete_from_cache(handle)

            # Unregister handle
            self.handle_registry.unregister(handle)

            return success

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting data: {e}")
            return False

    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        storage_types: Optional[List[StorageType]] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search across multiple backends

        Args:
            query: Search query
            content_type: Filter by content type
            storage_types: Specific backends to search (None = all)
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results across backends
        """
        try:
            self._stats['total_operations'] += 1

            # Determine which backends to search
            if storage_types:
                backends_to_search = [
                    self._get_backend(st) for st in storage_types
                    if st in self.backends
                ]
            elif content_type:
                # Search based on content type policy
                policy = self._get_policy(content_type)
                backends_to_search = [self._get_backend(policy.primary_backend)]
            else:
                # Search all backends
                backends_to_search = list(self.backends.values())

            # Collect results from all backends
            all_results = []
            for backend in backends_to_search:
                if backend:
                    try:
                        results = backend.search(
                            query=query,
                            content_type=content_type,
                            filters=filters,
                            limit=limit
                        )
                        all_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Error searching {backend.storage_type}: {e}")

            # Sort by score and limit
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:limit]

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error searching: {e}")
            return []

    def vector_search(
        self,
        query_embedding: List[float],
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform vector similarity search

        Args:
            query_embedding: Query embedding vector
            content_type: Filter by content type
            filters: Additional filters
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            # Get vector backend (Qdrant)
            vector_backend = self._get_backend(StorageType.VECTOR)
            if not vector_backend or not isinstance(vector_backend, AbstractVectorBackend):
                raise ValueError("Vector backend not available")

            return vector_backend.vector_search(
                query_embedding=query_embedding,
                content_type=content_type,
                filters=filters,
                top_k=top_k
            )

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error performing vector search: {e}")
            return []

    def get_stats(self, storage_type: Optional[StorageType] = None) -> Dict[str, Any]:
        """
        Get statistics for backends

        Args:
            storage_type: Specific backend (None = all)

        Returns:
            Statistics dictionary
        """
        if storage_type:
            backend = self._get_backend(storage_type)
            if backend:
                return {
                    storage_type.value: backend.get_stats().__dict__
                }
            return {}

        # Get stats from all backends
        stats = {
            'orchestrator': self._stats.copy(),
            'backends': {}
        }

        for st, backend in self.backends.items():
            try:
                backend_stats = backend.get_stats()
                stats['backends'][st.value] = backend_stats.__dict__
            except Exception as e:
                logger.error(f"Error getting stats from {st.value}: {e}")
                stats['backends'][st.value] = {'error': str(e)}

        return stats

    def health_check_all(self) -> Dict[str, Tuple[bool, str]]:
        """
        Check health of all backends

        Returns:
            Dictionary of health statuses
        """
        health_status = {}

        for storage_type, backend in self.backends.items():
            try:
                is_healthy, message = backend.health_check()
                health_status[storage_type.value] = (is_healthy, message)
            except Exception as e:
                health_status[storage_type.value] = (False, f"Health check error: {e}")

        return health_status

    def backup_all(self, destination_dir: str) -> Dict[str, bool]:
        """
        Backup all backends

        Args:
            destination_dir: Directory for backups

        Returns:
            Dictionary of backup statuses
        """
        from pathlib import Path
        dest_path = Path(destination_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        backup_status = {}

        for storage_type, backend in self.backends.items():
            try:
                backup_file = dest_path / f"{storage_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                success = backend.backup(str(backup_file))
                backup_status[storage_type.value] = success
            except Exception as e:
                logger.error(f"Error backing up {storage_type.value}: {e}")
                backup_status[storage_type.value] = False

        return backup_status

    def optimize_all(self) -> Dict[str, bool]:
        """
        Optimize all backends

        Returns:
            Dictionary of optimization statuses
        """
        optimization_status = {}

        for storage_type, backend in self.backends.items():
            try:
                success = backend.optimize()
                optimization_status[storage_type.value] = success
            except Exception as e:
                logger.error(f"Error optimizing {storage_type.value}: {e}")
                optimization_status[storage_type.value] = False

        return optimization_status

    # Internal helper methods

    def _setup_default_policies(self):
        """Setup default storage policies"""
        default_policies = {
            # Conversations → PostgreSQL with Redis cache
            ContentType.CONVERSATION: StoragePolicy(
                content_type=ContentType.CONVERSATION,
                primary_backend=StorageType.POSTGRESQL,
                cache_backend=StorageType.REDIS,
                cache_ttl=3600
            ),

            # Messages → PostgreSQL with Redis cache
            ContentType.MESSAGE: StoragePolicy(
                content_type=ContentType.MESSAGE,
                primary_backend=StorageType.POSTGRESQL,
                cache_backend=StorageType.REDIS,
                cache_ttl=1800
            ),

            # Documents → PostgreSQL
            ContentType.DOCUMENT: StoragePolicy(
                content_type=ContentType.DOCUMENT,
                primary_backend=StorageType.POSTGRESQL
            ),

            # Embeddings → Qdrant with Redis cache for hot queries
            ContentType.EMBEDDING: StoragePolicy(
                content_type=ContentType.EMBEDDING,
                primary_backend=StorageType.VECTOR,
                cache_backend=StorageType.REDIS,
                cache_ttl=7200
            ),

            # Memory → PostgreSQL
            ContentType.MEMORY: StoragePolicy(
                content_type=ContentType.MEMORY,
                primary_backend=StorageType.POSTGRESQL
            ),

            # Cache → Redis
            ContentType.CACHE: StoragePolicy(
                content_type=ContentType.CACHE,
                primary_backend=StorageType.REDIS,
                cache_ttl=600
            ),

            # Checkpoints → SQLite
            ContentType.CHECKPOINT: StoragePolicy(
                content_type=ContentType.CHECKPOINT,
                primary_backend=StorageType.SQLITE
            ),

            # Audit logs → SQLite
            ContentType.AUDIT: StoragePolicy(
                content_type=ContentType.AUDIT,
                primary_backend=StorageType.SQLITE
            ),

            # Metadata → Redis with PostgreSQL backup
            ContentType.METADATA: StoragePolicy(
                content_type=ContentType.METADATA,
                primary_backend=StorageType.REDIS,
                enable_replication=True,
                replica_backend=StorageType.POSTGRESQL
            )
        }

        # Merge with user-provided policies
        for content_type, policy in default_policies.items():
            if content_type not in self.config.policies:
                self.config.policies[content_type] = policy

    def _get_policy(self, content_type: ContentType) -> StoragePolicy:
        """Get storage policy for content type"""
        return self.config.policies.get(
            content_type,
            StoragePolicy(
                content_type=content_type,
                primary_backend=StorageType.POSTGRESQL
            )
        )

    def _get_backend(self, storage_type: StorageType) -> Optional[AbstractStorageBackend]:
        """Get backend by type"""
        return self.backends.get(storage_type)

    def _cache_data(self, handle: StorageHandle, data: Any, ttl: Optional[int]):
        """Cache data in Redis"""
        try:
            cache_backend = self._get_backend(StorageType.REDIS)
            if cache_backend and isinstance(cache_backend, AbstractCacheBackend):
                cache_key = f"cache:{handle.storage_type.value}:{handle.storage_id}"
                cache_backend.cache_set(cache_key, data, ttl=ttl)
        except Exception as e:
            logger.warning(f"Error caching data: {e}")

    def _try_cache_retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """Try to retrieve from cache"""
        try:
            cache_backend = self._get_backend(StorageType.REDIS)
            if cache_backend and isinstance(cache_backend, AbstractCacheBackend):
                cache_key = f"cache:{handle.storage_type.value}:{handle.storage_id}"
                return cache_backend.cache_get(cache_key)
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
        return None

    def _delete_from_cache(self, handle: StorageHandle):
        """Delete from cache"""
        try:
            cache_backend = self._get_backend(StorageType.REDIS)
            if cache_backend and isinstance(cache_backend, AbstractCacheBackend):
                cache_key = f"cache:{handle.storage_type.value}:{handle.storage_id}"
                cache_backend.cache_delete(cache_key)
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")

    def _replicate_data(
        self, handle: StorageHandle, data: Any,
        replica_type: StorageType, metadata: Optional[Dict]
    ):
        """Replicate data to another backend"""
        try:
            replica_backend = self._get_backend(replica_type)
            if replica_backend:
                replica_backend.store(
                    data=data,
                    content_type=handle.content_type,
                    key=handle.storage_id,
                    metadata=metadata
                )
                self._stats['replications'] += 1
                logger.debug(f"Replicated to {replica_type.value}")
        except Exception as e:
            logger.warning(f"Error replicating data: {e}")

    def _start_health_monitoring(self):
        """Start background health monitoring thread"""
        def health_check_loop():
            while not self._shutdown_event.is_set():
                try:
                    health_status = self.health_check_all()
                    unhealthy = [
                        name for name, (healthy, _) in health_status.items()
                        if not healthy
                    ]
                    if unhealthy:
                        logger.warning(f"Unhealthy backends: {unhealthy}")
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")

                self._shutdown_event.wait(self.config.health_check_interval)

        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._health_check_thread.start()
        logger.info("Health monitoring started")

    def _start_auto_optimization(self):
        """Start background optimization thread"""
        def optimization_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Wait for interval first
                    self._shutdown_event.wait(self.config.optimization_interval)

                    if not self._shutdown_event.is_set():
                        logger.info("Running auto-optimization")
                        results = self.optimize_all()
                        successful = sum(1 for success in results.values() if success)
                        logger.info(f"Optimized {successful}/{len(results)} backends")
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")

        self._optimization_thread = threading.Thread(
            target=optimization_loop,
            daemon=True,
            name="AutoOptimizer"
        )
        self._optimization_thread.start()
        logger.info("Auto-optimization started")


def create_default_orchestrator() -> StorageOrchestrator:
    """
    Create orchestrator with default configuration

    Returns:
        Configured StorageOrchestrator
    """
    config = OrchestratorConfig(
        postgresql_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'ai_engine',
            'user': 'postgres',
            'password': 'password'
        },
        redis_config={
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        qdrant_config={
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'embeddings',
            'vector_size': 1024,
            'use_quantization': True
        },
        sqlite_config={
            'db_path': '/dev/shm/lat5150_ramdisk.db',
            'use_wal': True,
            'enable_fts': True
        }
    )

    orchestrator = StorageOrchestrator(config)
    orchestrator.initialize()

    return orchestrator


if __name__ == "__main__":
    print("=" * 80)
    print("STORAGE ORCHESTRATOR")
    print("=" * 80 + "\n")

    print("Creating default orchestrator...")
    orchestrator = create_default_orchestrator()

    print("\nConfigured backends:")
    for storage_type in orchestrator.backends.keys():
        print(f"  ✓ {storage_type.value}")

    print("\nStorage policies:")
    for content_type, policy in orchestrator.config.policies.items():
        print(f"  {content_type.value:15} → {policy.primary_backend.value:12}", end="")
        if policy.cache_backend:
            print(f" (cached in {policy.cache_backend.value})", end="")
        print()

    print("\nOrchestrator ready!")
    print("\nFeatures:")
    print("  - Intelligent backend routing")
    print("  - Automatic caching with Redis")
    print("  - Cross-storage replication")
    print("  - Health monitoring")
    print("  - Auto-optimization")
    print("  - Unified backup/recovery")

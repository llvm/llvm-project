#!/usr/bin/env python3
"""
Redis Storage Backend Adapter

Provides unified interface for Redis cache/memory in the LAT5150DRVMIL AI engine.
Handles:
- Fast caching layer
- Session storage
- Temporary data with TTL
- Pub/sub for real-time updates
- Distributed locks
"""

import logging
import json
import redis
from redis.connection import ConnectionPool
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import time

from storage_abstraction import (
    AbstractCacheBackend,
    StorageType,
    ContentType,
    StorageTier,
    StorageHandle,
    SearchResult,
    StorageStats
)

logger = logging.getLogger(__name__)


class RedisStorageBackend(AbstractCacheBackend):
    """
    Redis storage adapter for AI engine caching

    Supports:
    - High-performance key-value storage
    - Automatic expiration (TTL)
    - Pub/sub messaging
    - Distributed locking
    - Pattern-based search
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Redis backend

        Args:
            config: Configuration dictionary with:
                - host: Redis host
                - port: Redis port (default: 6379)
                - db: Database number (default: 0)
                - password: Password (optional)
                - max_connections: Pool size (default: 50)
                - decode_responses: Decode to strings (default: True)
                - key_prefix: Prefix for all keys (default: "lat5150:")
        """
        super().__init__(config)
        self.storage_type = StorageType.REDIS

        # Connection configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.password = config.get('password')
        self.max_connections = config.get('max_connections', 50)
        self.decode_responses = config.get('decode_responses', True)
        self.key_prefix = config.get('key_prefix', 'lat5150:')

        # Connection pool
        self.pool = None
        self.client = None

        # Performance tracking
        self._hit_count = 0
        self._miss_count = 0
        self._access_times = []
        self._max_access_history = 1000

    def connect(self) -> bool:
        """
        Establish connection to Redis

        Returns:
            True if connection successful
        """
        try:
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses
            )

            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            self.client.ping()
            info = self.client.info('server')
            logger.info(f"Connected to Redis {info['redis_version']}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Close Redis connections

        Returns:
            True if disconnection successful
        """
        try:
            if self.client:
                self.client.close()
            if self.pool:
                self.pool.disconnect()
            logger.info("Redis connection closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            return False

    def store(
        self,
        data: Any,
        content_type: ContentType,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> StorageHandle:
        """
        Store data in Redis

        Args:
            data: Data to store (will be JSON serialized)
            content_type: Type of content
            key: Cache key (auto-generated if not provided)
            ttl: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            StorageHandle for retrieving data
        """
        try:
            start_time = time.time()

            # Generate key if not provided
            if not key:
                key = f"{content_type.value}:{int(time.time() * 1000000)}"

            full_key = f"{self.key_prefix}{key}"

            # Serialize data
            if isinstance(data, (dict, list)):
                value = json.dumps(data)
            elif isinstance(data, str):
                value = data
            else:
                value = str(data)

            # Store metadata separately if provided
            if metadata:
                metadata_key = f"{full_key}:metadata"
                self.client.set(metadata_key, json.dumps(metadata))
                if ttl:
                    self.client.expire(metadata_key, ttl)

            # Store data with TTL
            if ttl:
                self.client.setex(full_key, ttl, value)
            else:
                self.client.set(full_key, value)

            # Track performance
            access_time = (time.time() - start_time) * 1000
            self._access_times.append(access_time)
            if len(self._access_times) > self._max_access_history:
                self._access_times.pop(0)

            self._stats['operations'] += 1
            self._stats['bytes_written'] += len(value.encode())

            return StorageHandle(
                storage_type=self.storage_type,
                storage_id=full_key,
                content_type=content_type,
                tier=StorageTier.HOT,  # Redis is always hot storage
                created_at=datetime.now(),
                metadata=metadata or {}
            )

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error storing data in Redis: {e}")
            raise

    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """
        Retrieve data by handle

        Args:
            handle: Storage handle from store()

        Returns:
            Retrieved data or None if not found/expired
        """
        try:
            start_time = time.time()

            value = self.client.get(handle.storage_id)

            # Track cache hits/misses
            if value is None:
                self._miss_count += 1
                return None

            self._hit_count += 1

            # Try to parse as JSON
            try:
                data = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                data = value

            # Track performance
            access_time = (time.time() - start_time) * 1000
            self._access_times.append(access_time)
            if len(self._access_times) > self._max_access_history:
                self._access_times.pop(0)

            self._stats['operations'] += 1
            self._stats['bytes_read'] += len(str(value).encode())

            return data

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error retrieving data from Redis: {e}")
            return None

    def delete(self, handle: StorageHandle) -> bool:
        """
        Delete data by handle

        Args:
            handle: Storage handle

        Returns:
            True if deletion successful
        """
        try:
            # Delete main key and metadata
            deleted = self.client.delete(handle.storage_id)
            self.client.delete(f"{handle.storage_id}:metadata")

            self._stats['operations'] += 1
            return deleted > 0

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting data from Redis: {e}")
            return False

    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for keys matching pattern

        Args:
            query: Search pattern (supports * and ? wildcards)
            content_type: Filter by content type
            filters: Additional filters (not used for Redis)
            limit: Maximum results

        Returns:
            List of search results
        """
        try:
            # Build search pattern
            if content_type:
                pattern = f"{self.key_prefix}{content_type.value}:*{query}*"
            else:
                pattern = f"{self.key_prefix}*{query}*"

            # Find matching keys
            matching_keys = []
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                matching_keys.extend(keys)
                if cursor == 0 or len(matching_keys) >= limit:
                    break

            matching_keys = matching_keys[:limit]

            # Retrieve values for matching keys
            results = []
            for key in matching_keys:
                value = self.client.get(key)
                if value:
                    try:
                        content = json.loads(value)
                    except:
                        content = value

                    # Get metadata if exists
                    metadata = {}
                    metadata_value = self.client.get(f"{key}:metadata")
                    if metadata_value:
                        try:
                            metadata = json.loads(metadata_value)
                        except:
                            pass

                    # Determine content type from key
                    key_without_prefix = key.replace(self.key_prefix, '')
                    ct = ContentType.CACHE
                    if ':' in key_without_prefix:
                        ct_str = key_without_prefix.split(':')[0]
                        try:
                            ct = ContentType(ct_str)
                        except:
                            pass

                    handle = StorageHandle(
                        storage_type=self.storage_type,
                        storage_id=key,
                        content_type=ct,
                        tier=StorageTier.HOT,
                        created_at=datetime.now(),
                        metadata=metadata
                    )

                    results.append(SearchResult(
                        handle=handle,
                        score=1.0,  # Redis doesn't provide relevance scores
                        content=content,
                        metadata=metadata
                    ))

            self._stats['operations'] += 1
            return results

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error searching Redis: {e}")
            return []

    def get_stats(self) -> StorageStats:
        """
        Get Redis storage statistics

        Returns:
            StorageStats object
        """
        try:
            info = self.client.info('memory')
            stats_info = self.client.info('stats')

            # Count keys
            total_items = self.client.dbsize()

            # Memory usage
            total_size = info.get('used_memory', 0)

            # Calculate cache hit rate
            total_accesses = self._hit_count + self._miss_count
            hit_rate = (
                self._hit_count / total_accesses
                if total_accesses > 0 else 0.0
            )

            # Average access time
            avg_access_time = (
                sum(self._access_times) / len(self._access_times)
                if self._access_times else 0.0
            )

            return StorageStats(
                storage_type=self.storage_type,
                total_items=total_items,
                total_size_bytes=total_size,
                avg_access_time_ms=avg_access_time,
                health_status="healthy",
                custom_metrics={
                    'hit_rate': hit_rate,
                    'hits': self._hit_count,
                    'misses': self._miss_count,
                    'operations': self._stats['operations'],
                    'bytes_written': self._stats['bytes_written'],
                    'bytes_read': self._stats['bytes_read'],
                    'errors': self._stats['errors'],
                    'connected_clients': stats_info.get('connected_clients', 0),
                    'total_commands_processed': stats_info.get('total_commands_processed', 0)
                }
            )

        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return StorageStats(
                storage_type=self.storage_type,
                total_items=0,
                total_size_bytes=0,
                avg_access_time_ms=0,
                health_status="error"
            )

    def health_check(self) -> Tuple[bool, str]:
        """
        Check Redis health

        Returns:
            (is_healthy, status_message)
        """
        try:
            response = self.client.ping()
            if response:
                return (True, "Redis connection healthy")
            else:
                return (False, "Redis ping failed")
        except Exception as e:
            return (False, f"Redis health check failed: {e}")

    # Cache-specific methods (from AbstractCacheBackend)

    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        full_key = f"{self.key_prefix}{key}"
        handle = StorageHandle(
            storage_type=self.storage_type,
            storage_id=full_key,
            content_type=ContentType.CACHE,
            tier=StorageTier.HOT,
            created_at=datetime.now(),
            metadata={}
        )
        return self.retrieve(handle)

    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached value

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        try:
            self.store(
                data=value,
                content_type=ContentType.CACHE,
                key=key,
                ttl=ttl
            )
            return True
        except:
            return False

    def cache_delete(self, key: str) -> bool:
        """
        Delete cached value

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        full_key = f"{self.key_prefix}{key}"
        handle = StorageHandle(
            storage_type=self.storage_type,
            storage_id=full_key,
            content_type=ContentType.CACHE,
            tier=StorageTier.HOT,
            created_at=datetime.now(),
            metadata={}
        )
        return self.delete(handle)

    def cache_exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        full_key = f"{self.key_prefix}{key}"
        return self.client.exists(full_key) > 0

    def cache_clear(self) -> bool:
        """
        Clear all cached data with prefix

        Returns:
            True if successful
        """
        try:
            # Find all keys with prefix
            pattern = f"{self.key_prefix}*"
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=1000)
                if keys:
                    deleted_count += self.client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted_count} keys from Redis cache")
            return True

        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False

    def get_cache_hit_rate(self) -> float:
        """
        Get cache hit rate

        Returns:
            Hit rate as percentage (0.0-1.0)
        """
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    # Advanced Redis features

    def acquire_lock(self, lock_name: str, timeout: int = 10, blocking: bool = True) -> Optional[Any]:
        """
        Acquire distributed lock

        Args:
            lock_name: Name of lock
            timeout: Lock timeout in seconds
            blocking: Wait for lock if True

        Returns:
            Lock object or None if failed
        """
        try:
            lock_key = f"{self.key_prefix}lock:{lock_name}"
            lock = self.client.lock(lock_key, timeout=timeout, blocking=blocking)
            if lock.acquire(blocking=blocking):
                return lock
            return None
        except Exception as e:
            logger.error(f"Error acquiring Redis lock: {e}")
            return None

    def release_lock(self, lock) -> bool:
        """
        Release distributed lock

        Args:
            lock: Lock object from acquire_lock()

        Returns:
            True if successful
        """
        try:
            lock.release()
            return True
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
            return False

    def publish(self, channel: str, message: Any) -> int:
        """
        Publish message to channel

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received message
        """
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            return self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Error publishing to Redis channel: {e}")
            return 0

    def subscribe(self, *channels: str):
        """
        Subscribe to channels

        Args:
            channels: Channel names to subscribe to

        Returns:
            PubSub object for receiving messages
        """
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Error subscribing to Redis channels: {e}")
            return None

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Atomically increment counter

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New value after increment
        """
        full_key = f"{self.key_prefix}{key}"
        return self.client.incrby(full_key, amount)

    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Atomically decrement counter

        Args:
            key: Counter key
            amount: Amount to decrement

        Returns:
            New value after decrement
        """
        full_key = f"{self.key_prefix}{key}"
        return self.client.decrby(full_key, amount)

    def set_expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on existing key

        Args:
            key: Key to expire
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        full_key = f"{self.key_prefix}{key}"
        return self.client.expire(full_key, ttl)

    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key

        Args:
            key: Cache key

        Returns:
            Remaining seconds (-1 if no expiry, -2 if not exists)
        """
        full_key = f"{self.key_prefix}{key}"
        return self.client.ttl(full_key)

    def backup(self, destination: str) -> bool:
        """
        Trigger Redis BGSAVE for backup

        Args:
            destination: Not used (Redis saves to configured dir)

        Returns:
            True if backup started
        """
        try:
            self.client.bgsave()
            logger.info("Redis BGSAVE triggered")
            return True
        except Exception as e:
            logger.error(f"Error triggering Redis backup: {e}")
            return False

    def optimize(self) -> bool:
        """
        Optimize Redis memory (delete expired keys)

        Returns:
            True if successful
        """
        try:
            # Trigger active expiration
            info = self.client.info('stats')
            expired_keys_before = info.get('expired_keys', 0)

            # Force scan to trigger expiration
            cursor = 0
            while True:
                cursor, _ = self.client.scan(cursor, count=1000)
                if cursor == 0:
                    break

            info = self.client.info('stats')
            expired_keys_after = info.get('expired_keys', 0)

            expired_count = expired_keys_after - expired_keys_before
            logger.info(f"Redis optimization completed. Expired {expired_count} keys")
            return True

        except Exception as e:
            logger.error(f"Error optimizing Redis: {e}")
            return False


if __name__ == "__main__":
    print("=" * 80)
    print("REDIS STORAGE BACKEND")
    print("=" * 80 + "\n")

    # Example configuration
    config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'max_connections': 50,
        'key_prefix': 'lat5150:'
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("✓ Redis Storage Backend initialized")
    print("\nSupports:")
    print("  - High-performance caching")
    print("  - Automatic expiration (TTL)")
    print("  - Distributed locking")
    print("  - Pub/sub messaging")
    print("  - Atomic counters")
    print("  - Pattern-based search")
    print("  - Cache hit/miss tracking")

#!/usr/bin/env python3
"""
Cache Manager Module
--------------------
Provides comprehensive caching capabilities for the AI engine to improve
performance by avoiding redundant computations and API calls.

Features:
- In-memory LRU cache with size limits
- TTL (time-to-live) expiration
- Persistent caching to disk
- Cache statistics and hit/miss tracking
- Thread-safe operations
- Decorators for easy integration
- Cache warming/preloading
- Automatic cache invalidation
"""

import os
import json
import time
import hashlib
import pickle
import threading
import functools
import logging
from typing import Any, Optional, Callable, Dict, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Seconds until expiration
    hits: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def access(self):
        """Record a cache hit"""
        self.hits += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size_bytes': self.size_bytes,
            'size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'entry_count': self.entry_count,
            'hit_rate': round(self.hit_rate * 100, 2)
        }


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Implements Least Recently Used eviction policy when max_size is exceeded.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Record hit
            entry.access()
            self._stats.hits += 1

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default)
        """
        # Estimate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 1024  # Default estimate

        with self._lock:
            # Check if key already exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
                size_bytes=size_bytes
            )

            # Check memory limit
            while (self._stats.size_bytes + size_bytes > self.max_memory_bytes and
                   len(self._cache) > 0):
                self._evict_lru()

            # Check size limit
            while len(self._cache) >= self.max_size and len(self._cache) > 0:
                self._evict_lru()

            # Add entry
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes

            if key not in self._cache or key == list(self._cache.keys())[-1]:
                self._stats.entry_count = len(self._cache)

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return

        key, entry = self._cache.popitem(last=False)  # Remove oldest
        self._stats.size_bytes -= entry.size_bytes
        self._stats.evictions += 1
        self._stats.entry_count -= 1
        logger.debug(f"Evicted cache entry: {key}")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate (remove) a cache entry.

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
            logger.info("Cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._stats.size_bytes,
                entry_count=len(self._cache)
            )

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.size_bytes -= entry.size_bytes
                self._stats.evictions += 1

            self._stats.entry_count = len(self._cache)
            return len(expired_keys)


class PersistentCache:
    """
    Persistent cache with disk storage.

    Combines in-memory LRU cache with persistent disk storage for durability.
    """

    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory for cache files
            max_size: Maximum number of in-memory entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache = LRUCache(max_size, max_memory_mb, default_ttl)
        self._lock = threading.Lock()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Hash key to create filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory or disk)"""
        # Try memory cache first
        value = self._memory_cache.get(key)
        if value is not None:
            return value

        # Try disk cache
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    entry_data = pickle.load(f)

                entry = CacheEntry(**entry_data)

                # Check expiration
                if entry.is_expired():
                    file_path.unlink()
                    return None

                # Load into memory cache
                self._memory_cache.put(key, entry.value, entry.ttl)

                return entry.value
            except Exception as e:
                logger.error(f"Error loading cache from disk: {e}")
                file_path.unlink()  # Remove corrupted cache file

        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None, persist: bool = True):
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            persist: Whether to persist to disk
        """
        # Put in memory cache
        self._memory_cache.put(key, value, ttl)

        # Persist to disk if requested
        if persist:
            file_path = self._get_file_path(key)
            try:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl if ttl is not None else self._memory_cache.default_ttl
                )

                with open(file_path, 'wb') as f:
                    pickle.dump(entry.__dict__, f)
            except Exception as e:
                logger.error(f"Error persisting cache to disk: {e}")

    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry (memory and disk)"""
        # Remove from memory
        memory_removed = self._memory_cache.invalidate(key)

        # Remove from disk
        file_path = self._get_file_path(key)
        disk_removed = False
        if file_path.exists():
            file_path.unlink()
            disk_removed = True

        return memory_removed or disk_removed

    def clear(self):
        """Clear all cache (memory and disk)"""
        self._memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

        logger.info("Persistent cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        memory_stats = self._memory_cache.get_stats().to_dict()

        # Count disk entries
        disk_entries = len(list(self.cache_dir.glob("*.cache")))
        disk_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))

        return {
            'memory': memory_stats,
            'disk': {
                'entries': disk_entries,
                'size_bytes': disk_size,
                'size_mb': round(disk_size / (1024 * 1024), 2)
            }
        }


class CacheManager:
    """
    Global cache manager with multiple named caches.

    Provides centralized management of different cache instances for
    different purposes (API responses, model outputs, etc.).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for persistent caches
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/lat5150drvmil"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._caches: Dict[str, PersistentCache] = {}
        self._lock = threading.Lock()

    def get_cache(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None
    ) -> PersistentCache:
        """
        Get or create a named cache.

        Args:
            name: Cache name
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds

        Returns:
            Cache instance
        """
        with self._lock:
            if name not in self._caches:
                cache_dir = self.cache_dir / name
                self._caches[name] = PersistentCache(
                    cache_dir=str(cache_dir),
                    max_size=max_size,
                    max_memory_mb=max_memory_mb,
                    default_ttl=default_ttl
                )

            return self._caches[name]

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all caches"""
        with self._lock:
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }

    def clear_all(self):
        """Clear all caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()


# Global cache manager
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get global cache manager"""
    return _cache_manager


# Decorator for caching function results

def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results.

    Usage:
        @cached(cache_name="api_responses", ttl=3600)
        def call_api(endpoint, params):
            # ... expensive API call ...
            return response
    """
    def decorator(func: Callable) -> Callable:
        cache = get_cache_manager().get_cache(cache_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash of function name and arguments
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()

            # Try cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return result

            # Execute function
            logger.debug(f"Cache miss: {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Example 1: Using decorator
    @cached(cache_name="examples", ttl=60)
    def expensive_operation(x: int) -> int:
        print(f"Computing {x}...")
        time.sleep(1)
        return x * 2

    # Example 2: Using cache manager directly
    manager = get_cache_manager()
    api_cache = manager.get_cache("api_responses", ttl=300)

    # Cache API response
    api_cache.put("endpoint1", {"data": "response"})

    # Retrieve
    response = api_cache.get("endpoint1")
    print(f"Cached response: {response}")

    # Use cached function
    result1 = expensive_operation(5)  # Cache miss, takes 1s
    result2 = expensive_operation(5)  # Cache hit, instant

    # Get stats
    stats = manager.get_all_stats()
    print(json.dumps(stats, indent=2))

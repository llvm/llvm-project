"""
NCS2 Memory Pool Manager
=========================
Advanced memory pooling system for Intel NCS2 devices, enabling efficient
data management, graph caching, and high-throughput edge processing.

Features:
- Zero-copy memory pools for each device
- Graph/model caching to avoid reloads
- Pre-allocated inference buffers
- Memory-mapped I/O for large datasets
- Batch processing with memory reuse
- FIFO queue management
- Memory usage tracking and optimization

Author: LAT5150DRVMIL AI Platform
"""

import logging
import mmap
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryBuffer:
    """Represents a pre-allocated memory buffer."""
    buffer_id: int
    size_bytes: int
    data: np.ndarray
    is_locked: bool = False
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0


@dataclass
class GraphCache:
    """Cached compiled graph/model."""
    graph_id: str
    graph_data: bytes
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    size_bytes: int
    load_count: int = 0
    last_used: float = field(default_factory=time.time)


class NCS2MemoryPool:
    """
    Memory pool manager for a single NCS2 device.

    Provides efficient memory management with:
    - Pre-allocated buffers for zero-copy operations
    - Graph caching to avoid repeated loads
    - Memory-mapped I/O for large datasets
    - Automatic buffer recycling
    """

    def __init__(
        self,
        device_id: int,
        pool_size_mb: int = 512,
        max_cached_graphs: int = 10
    ):
        """
        Initialize memory pool for a device.

        Args:
            device_id: NCS2 device ID
            pool_size_mb: Total memory pool size in MB
            max_cached_graphs: Maximum number of cached graphs
        """
        self.device_id = device_id
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.max_cached_graphs = max_cached_graphs

        # Memory buffers
        self.buffers: Dict[int, MemoryBuffer] = {}
        self.buffer_queue = deque()  # Available buffers
        self.next_buffer_id = 0

        # Graph cache
        self.graph_cache: Dict[str, GraphCache] = {}

        # Memory-mapped files
        self.mmaps: Dict[str, mmap.mmap] = {}

        # Statistics
        self.total_allocated_bytes = 0
        self.peak_memory_bytes = 0
        self.buffer_hits = 0
        self.buffer_misses = 0
        self.graph_cache_hits = 0
        self.graph_cache_misses = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            f"NCS2 Memory Pool initialized for device {device_id}: "
            f"{pool_size_mb}MB, max {max_cached_graphs} cached graphs"
        )

    def allocate_buffer(
        self,
        size_bytes: int,
        dtype: np.dtype = np.float32
    ) -> Optional[MemoryBuffer]:
        """
        Allocate a memory buffer.

        Args:
            size_bytes: Buffer size in bytes
            dtype: NumPy data type

        Returns:
            MemoryBuffer or None if allocation fails
        """
        with self.lock:
            # Check if we have space
            if self.total_allocated_bytes + size_bytes > self.pool_size_bytes:
                logger.warning(
                    f"Device {self.device_id}: Pool full "
                    f"({self.total_allocated_bytes}/{self.pool_size_bytes} bytes)"
                )
                # Try to free unused buffers
                self._free_unused_buffers()

                if self.total_allocated_bytes + size_bytes > self.pool_size_bytes:
                    return None

            # Allocate buffer
            buffer = MemoryBuffer(
                buffer_id=self.next_buffer_id,
                size_bytes=size_bytes,
                data=np.empty(size_bytes // dtype.itemsize, dtype=dtype)
            )

            self.buffers[buffer.buffer_id] = buffer
            self.next_buffer_id += 1
            self.total_allocated_bytes += size_bytes

            if self.total_allocated_bytes > self.peak_memory_bytes:
                self.peak_memory_bytes = self.total_allocated_bytes

            logger.debug(
                f"Device {self.device_id}: Allocated buffer {buffer.buffer_id} "
                f"({size_bytes} bytes)"
            )

            return buffer

    def get_buffer(
        self,
        size_bytes: int,
        dtype: np.dtype = np.float32
    ) -> Optional[MemoryBuffer]:
        """
        Get a buffer from pool or allocate new one.

        Args:
            size_bytes: Required buffer size
            dtype: NumPy data type

        Returns:
            MemoryBuffer or None if allocation fails
        """
        with self.lock:
            # Try to find available buffer of exact size
            for buffer in list(self.buffer_queue):
                if buffer.size_bytes == size_bytes and not buffer.is_locked:
                    self.buffer_queue.remove(buffer)
                    buffer.is_locked = True
                    buffer.last_used = time.time()
                    buffer.usage_count += 1
                    self.buffer_hits += 1
                    return buffer

            # No suitable buffer found, allocate new one
            self.buffer_misses += 1
            return self.allocate_buffer(size_bytes, dtype)

    def release_buffer(self, buffer: MemoryBuffer):
        """
        Release buffer back to pool.

        Args:
            buffer: Buffer to release
        """
        with self.lock:
            if buffer.buffer_id not in self.buffers:
                logger.warning(f"Buffer {buffer.buffer_id} not in pool")
                return

            buffer.is_locked = False
            buffer.last_used = time.time()
            self.buffer_queue.append(buffer)

            logger.debug(
                f"Device {self.device_id}: Released buffer {buffer.buffer_id}"
            )

    def free_buffer(self, buffer_id: int):
        """
        Free a buffer permanently.

        Args:
            buffer_id: Buffer ID to free
        """
        with self.lock:
            if buffer_id not in self.buffers:
                return

            buffer = self.buffers[buffer_id]
            self.total_allocated_bytes -= buffer.size_bytes

            # Remove from queue if present
            if buffer in self.buffer_queue:
                self.buffer_queue.remove(buffer)

            del self.buffers[buffer_id]

            logger.debug(
                f"Device {self.device_id}: Freed buffer {buffer_id}"
            )

    def _free_unused_buffers(self):
        """Free buffers that haven't been used recently."""
        with self.lock:
            current_time = time.time()
            freed_count = 0

            # Free unlocked buffers not used in last 60 seconds
            for buffer_id in list(self.buffers.keys()):
                buffer = self.buffers[buffer_id]
                if not buffer.is_locked and (current_time - buffer.last_used) > 60:
                    self.free_buffer(buffer_id)
                    freed_count += 1

            if freed_count > 0:
                logger.info(
                    f"Device {self.device_id}: Freed {freed_count} unused buffers"
                )

    def cache_graph(
        self,
        graph_id: str,
        graph_data: bytes,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]]
    ) -> bool:
        """
        Cache a compiled graph.

        Args:
            graph_id: Unique graph identifier
            graph_data: Compiled graph blob
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes

        Returns:
            True if cached successfully
        """
        with self.lock:
            # Check if already cached
            if graph_id in self.graph_cache:
                cache_entry = self.graph_cache[graph_id]
                cache_entry.load_count += 1
                cache_entry.last_used = time.time()
                return True

            # Check cache size
            if len(self.graph_cache) >= self.max_cached_graphs:
                # Evict least recently used graph
                oldest_id = min(
                    self.graph_cache.keys(),
                    key=lambda k: self.graph_cache[k].last_used
                )
                logger.debug(
                    f"Device {self.device_id}: Evicting graph {oldest_id} from cache"
                )
                del self.graph_cache[oldest_id]

            # Cache graph
            cache_entry = GraphCache(
                graph_id=graph_id,
                graph_data=graph_data,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                size_bytes=len(graph_data)
            )

            self.graph_cache[graph_id] = cache_entry

            logger.info(
                f"Device {self.device_id}: Cached graph {graph_id} "
                f"({len(graph_data)} bytes)"
            )

            return True

    def get_cached_graph(self, graph_id: str) -> Optional[GraphCache]:
        """
        Get cached graph.

        Args:
            graph_id: Graph identifier

        Returns:
            GraphCache or None if not found
        """
        with self.lock:
            if graph_id in self.graph_cache:
                cache_entry = self.graph_cache[graph_id]
                cache_entry.load_count += 1
                cache_entry.last_used = time.time()
                self.graph_cache_hits += 1
                return cache_entry

            self.graph_cache_misses += 1
            return None

    def mmap_file(self, file_path: str) -> Optional[mmap.mmap]:
        """
        Memory-map a file for zero-copy access.

        Args:
            file_path: Path to file

        Returns:
            mmap object or None if failed
        """
        try:
            if file_path in self.mmaps:
                return self.mmaps[file_path]

            # Open file and create memory map
            fd = os.open(file_path, os.O_RDONLY)
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

            self.mmaps[file_path] = mm

            logger.debug(
                f"Device {self.device_id}: Memory-mapped {file_path}"
            )

            return mm

        except Exception as e:
            logger.error(f"Failed to mmap {file_path}: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get memory pool statistics."""
        with self.lock:
            return {
                "device_id": self.device_id,
                "total_allocated_bytes": self.total_allocated_bytes,
                "peak_memory_bytes": self.peak_memory_bytes,
                "pool_size_bytes": self.pool_size_bytes,
                "utilization": self.total_allocated_bytes / self.pool_size_bytes,
                "buffer_count": len(self.buffers),
                "available_buffers": len(self.buffer_queue),
                "buffer_hits": self.buffer_hits,
                "buffer_misses": self.buffer_misses,
                "buffer_hit_rate": (
                    self.buffer_hits / (self.buffer_hits + self.buffer_misses)
                    if (self.buffer_hits + self.buffer_misses) > 0 else 0.0
                ),
                "cached_graphs": len(self.graph_cache),
                "graph_cache_hits": self.graph_cache_hits,
                "graph_cache_misses": self.graph_cache_misses,
                "graph_cache_hit_rate": (
                    self.graph_cache_hits / (self.graph_cache_hits + self.graph_cache_misses)
                    if (self.graph_cache_hits + self.graph_cache_misses) > 0 else 0.0
                )
            }

    def reset_stats(self):
        """Reset statistics counters."""
        with self.lock:
            self.buffer_hits = 0
            self.buffer_misses = 0
            self.graph_cache_hits = 0
            self.graph_cache_misses = 0

    def clear(self):
        """Clear all memory pools and caches."""
        with self.lock:
            # Free all buffers
            self.buffers.clear()
            self.buffer_queue.clear()
            self.total_allocated_bytes = 0

            # Clear graph cache
            self.graph_cache.clear()

            # Close memory maps
            for mm in self.mmaps.values():
                mm.close()
            self.mmaps.clear()

            logger.info(f"Device {self.device_id}: Memory pool cleared")


class NCS2MemoryPoolManager:
    """
    Manages memory pools for multiple NCS2 devices.

    Provides unified interface for memory management across all devices.
    """

    def __init__(
        self,
        device_count: int = 3,
        pool_size_mb_per_device: int = 512,
        max_cached_graphs_per_device: int = 10
    ):
        """
        Initialize memory pool manager.

        Args:
            device_count: Number of NCS2 devices (1-3)
            pool_size_mb_per_device: Memory pool size per device
            max_cached_graphs_per_device: Max cached graphs per device
        """
        self.device_count = min(device_count, 3)  # Support up to 3 devices
        self.pools: Dict[int, NCS2MemoryPool] = {}

        # Create pool for each device
        for device_id in range(self.device_count):
            self.pools[device_id] = NCS2MemoryPool(
                device_id=device_id,
                pool_size_mb=pool_size_mb_per_device,
                max_cached_graphs=max_cached_graphs_per_device
            )

        logger.info(
            f"NCS2 Memory Pool Manager initialized: {self.device_count} device(s), "
            f"{pool_size_mb_per_device}MB per device"
        )

    def get_pool(self, device_id: int) -> Optional[NCS2MemoryPool]:
        """Get memory pool for a specific device."""
        return self.pools.get(device_id)

    def get_total_memory_mb(self) -> float:
        """Get total memory across all pools."""
        total_bytes = sum(pool.pool_size_bytes for pool in self.pools.values())
        return total_bytes / (1024 * 1024)

    def get_allocated_memory_mb(self) -> float:
        """Get total allocated memory across all pools."""
        total_bytes = sum(pool.total_allocated_bytes for pool in self.pools.values())
        return total_bytes / (1024 * 1024)

    def get_utilization(self) -> float:
        """Get average memory utilization across all devices."""
        if not self.pools:
            return 0.0

        total_util = sum(
            pool.total_allocated_bytes / pool.pool_size_bytes
            for pool in self.pools.values()
        )
        return total_util / len(self.pools)

    def get_all_stats(self) -> Dict:
        """Get statistics for all devices."""
        return {
            "device_count": self.device_count,
            "total_memory_mb": self.get_total_memory_mb(),
            "allocated_memory_mb": self.get_allocated_memory_mb(),
            "average_utilization": self.get_utilization(),
            "devices": {
                device_id: pool.get_stats()
                for device_id, pool in self.pools.items()
            }
        }

    def clear_all(self):
        """Clear all memory pools."""
        for pool in self.pools.values():
            pool.clear()

        logger.info("All memory pools cleared")


# Singleton instance
_memory_pool_manager: Optional[NCS2MemoryPoolManager] = None


def get_memory_pool_manager(
    device_count: int = 3,
    pool_size_mb: int = 512
) -> NCS2MemoryPoolManager:
    """
    Get or create singleton memory pool manager.

    Args:
        device_count: Number of devices (1-3)
        pool_size_mb: Memory pool size per device

    Returns:
        NCS2MemoryPoolManager instance
    """
    global _memory_pool_manager

    if _memory_pool_manager is None:
        _memory_pool_manager = NCS2MemoryPoolManager(
            device_count=device_count,
            pool_size_mb_per_device=pool_size_mb
        )

    return _memory_pool_manager

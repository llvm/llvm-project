"""
Performance Package
-------------------
Comprehensive performance monitoring, profiling, and optimization utilities
for the LAT5150DRVMIL AI Engine.

Modules:
- gpu_check: Hardware acceleration detection
- performance_monitor: Real-time metrics and profiling
- cache_manager: Multi-level caching system

Usage:
    from performance import get_monitor, profile, cached
    from performance.gpu_check import HardwareDetector

    # Profile functions
    @profile
    def my_function():
        pass

    # Cache results
    @cached(cache_name="api", ttl=3600)
    def api_call():
        pass

    # Check hardware
    detector = HardwareDetector()
    caps = detector.detect_all()
"""

from .performance_monitor import (
    get_monitor,
    profile,
    profile_with_metrics,
    Timer,
    PerformanceMonitor,
    PerformanceMetric,
    FunctionProfile
)

from .cache_manager import (
    get_cache_manager,
    cached,
    CacheManager,
    PersistentCache,
    LRUCache,
    CacheEntry,
    CacheStats
)

__all__ = [
    # Performance monitoring
    'get_monitor',
    'profile',
    'profile_with_metrics',
    'Timer',
    'PerformanceMonitor',
    'PerformanceMetric',
    'FunctionProfile',

    # Caching
    'get_cache_manager',
    'cached',
    'CacheManager',
    'PersistentCache',
    'LRUCache',
    'CacheEntry',
    'CacheStats',
]

#!/usr/bin/env python3
"""
Performance Monitoring Module
------------------------------
Provides comprehensive performance monitoring, profiling, and metrics collection
for the LAT5150DRVMIL AI Engine.

Features:
- Real-time performance metrics collection
- Function-level profiling with decorators
- Memory usage tracking
- GPU utilization monitoring
- Custom metrics and timers
- Performance reporting and visualization
"""

import time
import functools
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


@dataclass
class FunctionProfile:
    """Profiling data for a function"""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: float = 0.0

    def update(self, elapsed: float):
        """Update profile with new timing"""
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.call_count
        self.last_call_time = elapsed

    def to_dict(self) -> Dict:
        return {
            'function': self.function_name,
            'calls': self.call_count,
            'total_time_s': round(self.total_time, 4),
            'avg_time_ms': round(self.avg_time * 1000, 2),
            'min_time_ms': round(self.min_time * 1000, 2),
            'max_time_ms': round(self.max_time * 1000, 2)
        }


class PerformanceMonitor:
    """
    Global performance monitoring singleton.

    Collects and aggregates performance metrics throughout the application.
    Thread-safe for concurrent access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._metrics: List[PerformanceMetric] = []
        self._function_profiles: Dict[str, FunctionProfile] = {}
        self._custom_timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._enabled = True
        self._lock = threading.Lock()

        logger.info("Performance monitor initialized")

    def is_enabled(self) -> bool:
        """Check if monitoring is enabled"""
        return self._enabled

    def enable(self):
        """Enable performance monitoring"""
        self._enabled = True
        logger.info("Performance monitoring enabled")

    def disable(self):
        """Disable performance monitoring"""
        self._enabled = False
        logger.info("Performance monitoring disabled")

    def record_metric(self, name: str, value: float, unit: str = "", **tags):
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement (e.g., 'ms', 'MB', 'tokens')
            **tags: Additional tags for categorization
        """
        if not self._enabled:
            return

        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags
        )

        with self._lock:
            self._metrics.append(metric)

    def record_function_call(self, function_name: str, elapsed: float):
        """Record function execution time"""
        if not self._enabled:
            return

        with self._lock:
            if function_name not in self._function_profiles:
                self._function_profiles[function_name] = FunctionProfile(function_name)

            self._function_profiles[function_name].update(elapsed)

    def start_timer(self, name: str):
        """Start a custom timer"""
        if not self._enabled:
            return

        with self._lock:
            self._custom_timers[name] = time.time()

    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop a custom timer and record elapsed time.

        Returns:
            Elapsed time in seconds, or None if timer not found
        """
        if not self._enabled:
            return None

        with self._lock:
            if name in self._custom_timers:
                start_time = self._custom_timers[name]
                elapsed = time.time() - start_time
                del self._custom_timers[name]

                self.record_metric(
                    name=f"timer.{name}",
                    value=elapsed * 1000,  # Convert to ms
                    unit="ms"
                )

                return elapsed

        return None

    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter"""
        if not self._enabled:
            return

        with self._lock:
            self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_function_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """Get profile for a specific function"""
        with self._lock:
            return self._function_profiles.get(function_name)

    def get_all_function_profiles(self) -> Dict[str, FunctionProfile]:
        """Get all function profiles"""
        with self._lock:
            return self._function_profiles.copy()

    def get_metrics(self, name: Optional[str] = None) -> List[PerformanceMetric]:
        """
        Get recorded metrics.

        Args:
            name: Optional filter by metric name

        Returns:
            List of metrics
        """
        with self._lock:
            if name:
                return [m for m in self._metrics if m.name == name]
            return self._metrics.copy()

    def get_summary(self) -> Dict:
        """Get summary of all performance data"""
        with self._lock:
            return {
                'metrics_count': len(self._metrics),
                'functions_profiled': len(self._function_profiles),
                'active_timers': len(self._custom_timers),
                'counters': dict(self._counters),
                'function_profiles': {
                    name: profile.to_dict()
                    for name, profile in self._function_profiles.items()
                }
            }

    def export_report(self, filepath: str):
        """Export performance report to JSON file"""
        summary = self.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        summary['metrics'] = [m.to_dict() for m in self._metrics[-100:]]  # Last 100 metrics

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Performance report exported to {filepath}")

    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("  PERFORMANCE SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overview:")
        print(f"   Metrics Recorded: {summary['metrics_count']}")
        print(f"   Functions Profiled: {summary['functions_profiled']}")
        print(f"   Active Timers: {summary['active_timers']}")

        if summary['counters']:
            print(f"\n🔢 Counters:")
            for name, value in sorted(summary['counters'].items()):
                print(f"   {name}: {value:,}")

        if summary['function_profiles']:
            print(f"\n⏱️  Top Functions by Total Time:")
            profiles = sorted(
                summary['function_profiles'].values(),
                key=lambda p: p['total_time_s'],
                reverse=True
            )
            for profile in profiles[:10]:  # Top 10
                print(f"   {profile['function']}")
                print(f"      Calls: {profile['calls']:,} | "
                      f"Avg: {profile['avg_time_ms']:.2f}ms | "
                      f"Total: {profile['total_time_s']:.2f}s")

        print("\n" + "=" * 80 + "\n")

    def reset(self):
        """Reset all performance data"""
        with self._lock:
            self._metrics.clear()
            self._function_profiles.clear()
            self._custom_timers.clear()
            self._counters.clear()

        logger.info("Performance monitor reset")


# Global instance
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _monitor


# Decorators for easy profiling

def profile(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.

    Usage:
        @profile
        def my_function():
            # ... code ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        if not monitor.is_enabled():
            return func(*args, **kwargs)

        function_name = f"{func.__module__}.{func.__name__}"
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            monitor.record_function_call(function_name, elapsed)

    return wrapper


def profile_with_metrics(metric_name: Optional[str] = None, **metric_tags):
    """
    Decorator to profile function and record as metric.

    Usage:
        @profile_with_metrics(metric_name="api_call", endpoint="gemini")
        def call_api():
            # ... code ...
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            if not monitor.is_enabled():
                return func(*args, **kwargs)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                elapsed = time.time() - start_time
                tags = metric_tags.copy()
                tags['success'] = str(success)

                monitor.record_metric(
                    name=name,
                    value=elapsed * 1000,  # Convert to ms
                    unit="ms",
                    **tags
                )

        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("my_operation"):
            # ... code to time ...
    """

    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor or get_monitor()
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.monitor.is_enabled():
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor.is_enabled() and self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.monitor.record_metric(
                name=self.name,
                value=self.elapsed * 1000,  # Convert to ms
                unit="ms",
                success=str(exc_type is None)
            )


# Example usage
if __name__ == "__main__":
    # Example 1: Using decorator
    @profile
    def example_function():
        time.sleep(0.1)
        return "done"

    # Example 2: Using context manager
    with Timer("data_processing"):
        time.sleep(0.05)

    # Example 3: Manual metrics
    monitor = get_monitor()
    monitor.record_metric("memory_usage", 1024.5, "MB")
    monitor.increment_counter("api_calls")

    # Example 4: Custom timer
    monitor.start_timer("complex_operation")
    time.sleep(0.02)
    elapsed = monitor.stop_timer("complex_operation")

    # Call example function multiple times
    for _ in range(5):
        example_function()

    # Print summary
    monitor.print_summary()

    # Export report
    monitor.export_report("performance_report.json")

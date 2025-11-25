#!/usr/bin/env python3
"""
Comprehensive Hook System for Auto-Optimization
Integrated from claude-backups framework

Features:
- Pre-query hooks (input validation, backend selection)
- Post-query hooks (performance logging, auto-optimization)
- Git hooks integration (pre-commit, post-commit, pre-push)
- Performance monitoring hooks
- Automatic optimization based on performance history
- NPU/GNA/P-core workload routing hooks
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict


class HookType(Enum):
    """Hook types"""
    PRE_QUERY = "pre_query"
    POST_QUERY = "post_query"
    PRE_COMMIT = "pre_commit"
    POST_COMMIT = "post_commit"
    PRE_PUSH = "pre_push"
    POST_PUSH = "post_push"
    PERFORMANCE_MONITOR = "performance_monitor"
    OPTIMIZATION = "optimization"
    ERROR_HANDLER = "error_handler"


class HookPriority(Enum):
    """Hook execution priority"""
    CRITICAL = 0     # Security, validation
    HIGH = 10        # Performance optimization
    NORMAL = 20      # Logging, monitoring
    LOW = 30         # Non-essential operations


@dataclass
class HookContext:
    """Context passed to hooks"""
    hook_type: HookType
    timestamp: float
    data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class HookResult:
    """Result from hook execution"""
    success: bool
    message: str = ""
    data: Dict[str, Any] = None
    modified_context: Optional[HookContext] = None
    abort: bool = False  # If True, abort the operation


class Hook:
    """Base hook class"""

    def __init__(
        self,
        name: str,
        hook_type: HookType,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True
    ):
        self.name = name
        self.hook_type = hook_type
        self.priority = priority
        self.enabled = enabled
        self.execution_count = 0
        self.total_time_ms = 0.0
        self.failure_count = 0

    def execute(self, context: HookContext) -> HookResult:
        """
        Execute hook

        Args:
            context: Hook context

        Returns:
            Hook result
        """
        if not self.enabled:
            return HookResult(success=True, message=f"Hook {self.name} disabled")

        start_time = time.time()

        try:
            result = self._run(context)
            self.execution_count += 1
            self.total_time_ms += (time.time() - start_time) * 1000
            return result
        except Exception as e:
            self.failure_count += 1
            return HookResult(
                success=False,
                message=f"Hook {self.name} failed: {str(e)}",
                abort=True
            )

    def _run(self, context: HookContext) -> HookResult:
        """Override in subclasses"""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics"""
        avg_time = self.total_time_ms / self.execution_count if self.execution_count > 0 else 0

        return {
            "name": self.name,
            "type": self.hook_type.value,
            "enabled": self.enabled,
            "executions": self.execution_count,
            "failures": self.failure_count,
            "avg_time_ms": avg_time,
            "success_rate": (self.execution_count - self.failure_count) / max(1, self.execution_count) * 100
        }


# Pre-Query Hooks

class InputValidationHook(Hook):
    """Validate input before processing"""

    def __init__(self):
        super().__init__(
            name="input_validation",
            hook_type=HookType.PRE_QUERY,
            priority=HookPriority.CRITICAL
        )

    def _run(self, context: HookContext) -> HookResult:
        """Validate input"""
        prompt = context.data.get("prompt", "")

        # Check for empty input
        if not prompt or not prompt.strip():
            return HookResult(
                success=False,
                message="Empty prompt",
                abort=True
            )

        # Check for excessively long input
        if len(prompt) > 100000:
            return HookResult(
                success=False,
                message=f"Prompt too long: {len(prompt)} characters (max 100000)",
                abort=True
            )

        # Security check: detect potential injection attempts
        dangerous_patterns = ["'; DROP TABLE", "../../", "<script>"]
        for pattern in dangerous_patterns:
            if pattern in prompt:
                return HookResult(
                    success=False,
                    message=f"Potentially dangerous input detected: {pattern}",
                    abort=True
                )

        return HookResult(success=True, message="Input validated")


class BackendSelectionHook(Hook):
    """Select optimal backend based on query characteristics"""

    def __init__(self, heterogeneous_executor=None):
        super().__init__(
            name="backend_selection",
            hook_type=HookType.PRE_QUERY,
            priority=HookPriority.HIGH
        )
        self.executor = heterogeneous_executor

    def _run(self, context: HookContext) -> HookResult:
        """Select optimal backend"""
        if not self.executor:
            return HookResult(success=True, message="No executor configured")

        prompt = context.data.get("prompt", "")

        # Analyze query to create workload profile
        is_realtime = "quick" in prompt.lower() or "fast" in prompt.lower()
        is_audio = "audio" in prompt.lower() or "voice" in prompt.lower() or "speech" in prompt.lower()
        is_continuous = "continuous" in prompt.lower() or "stream" in prompt.lower()

        # Estimate complexity
        complexity_score = min(1.0, len(prompt) / 1000 + (prompt.count(' ') / 100))

        # Estimate model size based on query
        model_size_mb = 50  # Default small model
        if "large" in prompt.lower() or "complex" in prompt.lower():
            model_size_mb = 500

        # Create workload profile
        from heterogeneous_executor import WorkloadProfile

        workload = WorkloadProfile(
            model_size_mb=model_size_mb,
            input_size_kb=len(prompt) / 1024,
            is_realtime=is_realtime,
            is_continuous=is_continuous,
            is_audio=is_audio,
            complexity_score=complexity_score,
            latency_requirement_ms=100 if is_realtime else 5000
        )

        # Select backend
        backend = self.executor.select_backend(workload)

        # Modify context to include backend selection
        context.data["selected_backend"] = backend.value
        context.data["workload_profile"] = asdict(workload)

        return HookResult(
            success=True,
            message=f"Selected backend: {backend.value}",
            modified_context=context
        )


# Post-Query Hooks

class PerformanceLoggingHook(Hook):
    """Log query performance"""

    def __init__(self, log_file: str = "performance_log.jsonl"):
        super().__init__(
            name="performance_logging",
            hook_type=HookType.POST_QUERY,
            priority=HookPriority.NORMAL
        )
        self.log_file = log_file

    def _run(self, context: HookContext) -> HookResult:
        """Log performance data"""
        log_entry = {
            "timestamp": context.timestamp,
            "prompt_hash": hashlib.sha256(
                context.data.get("prompt", "").encode()
            ).hexdigest()[:16],
            "backend": context.data.get("backend", "unknown"),
            "latency_ms": context.data.get("latency_ms", 0),
            "cached": context.data.get("cached", False),
            "success": context.data.get("success", False),
            "model": context.data.get("model", "unknown")
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            return HookResult(success=True, message="Performance logged")
        except Exception as e:
            return HookResult(success=False, message=f"Logging failed: {e}")


class AutoOptimizationHook(Hook):
    """Automatically optimize based on performance history"""

    def __init__(self):
        super().__init__(
            name="auto_optimization",
            hook_type=HookType.POST_QUERY,
            priority=HookPriority.HIGH
        )
        self.performance_history = []
        self.optimization_threshold = 10  # Optimize after 10 queries

    def _run(self, context: HookContext) -> HookResult:
        """Analyze performance and optimize"""
        latency_ms = context.data.get("latency_ms", 0)
        backend = context.data.get("backend", "unknown")
        success = context.data.get("success", False)

        # Record performance
        self.performance_history.append({
            "latency_ms": latency_ms,
            "backend": backend,
            "success": success,
            "timestamp": context.timestamp
        })

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Optimize if threshold reached
        if len(self.performance_history) >= self.optimization_threshold:
            optimizations = self._analyze_and_optimize()

            return HookResult(
                success=True,
                message=f"Optimizations applied: {len(optimizations)}",
                data={"optimizations": optimizations}
            )

        return HookResult(success=True, message="Performance recorded")

    def _analyze_and_optimize(self) -> List[str]:
        """Analyze performance and suggest optimizations"""
        optimizations = []

        # Group by backend
        by_backend = defaultdict(list)
        for record in self.performance_history:
            by_backend[record["backend"]].append(record["latency_ms"])

        # Find slow backends
        for backend, latencies in by_backend.items():
            avg_latency = sum(latencies) / len(latencies)

            if avg_latency > 5000:  # > 5 seconds
                optimizations.append(f"Backend {backend} is slow (avg {avg_latency:.0f}ms) - consider switching")

        # Check success rate
        success_rate = sum(1 for r in self.performance_history if r["success"]) / len(self.performance_history)

        if success_rate < 0.9:
            optimizations.append(f"Low success rate ({success_rate*100:.1f}%) - review error handling")

        return optimizations


# Git Hooks

class PreCommitHook(Hook):
    """Pre-commit validation and formatting"""

    def __init__(self):
        super().__init__(
            name="pre_commit",
            hook_type=HookType.PRE_COMMIT,
            priority=HookPriority.CRITICAL
        )

    def _run(self, context: HookContext) -> HookResult:
        """Run pre-commit checks"""
        files = context.data.get("files", [])

        # Check for large files
        large_files = []
        for file in files:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / (1024 * 1024)
                if size_mb > 100:  # 100 MB
                    large_files.append(f"{file} ({size_mb:.1f} MB)")

        if large_files:
            return HookResult(
                success=False,
                message=f"Large files detected: {', '.join(large_files)}",
                abort=True
            )

        # Check for sensitive data
        sensitive_patterns = [".env", "id_rsa", "password", "secret_key"]
        sensitive_files = []

        for file in files:
            if any(pattern in file.lower() for pattern in sensitive_patterns):
                sensitive_files.append(file)

        if sensitive_files:
            return HookResult(
                success=False,
                message=f"Potentially sensitive files: {', '.join(sensitive_files)}",
                abort=False  # Warning only
            )

        return HookResult(success=True, message="Pre-commit checks passed")


class HookManager:
    """Manage and execute hooks"""

    def __init__(self):
        self.hooks: Dict[HookType, List[Hook]] = defaultdict(list)
        self.global_enabled = True
        self._lock = threading.Lock()

        print("✓ Hook Manager initialized")

    def register_hook(self, hook: Hook):
        """Register a hook"""
        with self._lock:
            self.hooks[hook.hook_type].append(hook)
            # Sort by priority
            self.hooks[hook.hook_type].sort(key=lambda h: h.priority.value)

        print(f"  ✓ Registered hook: {hook.name} ({hook.hook_type.value}, priority {hook.priority.value})")

    def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext
    ) -> Tuple[bool, List[HookResult]]:
        """
        Execute all hooks of a given type

        Args:
            hook_type: Type of hooks to execute
            context: Hook context

        Returns:
            (success, results): Overall success and individual results
        """
        if not self.global_enabled:
            return (True, [])

        results = []
        current_context = context

        with self._lock:
            hooks = self.hooks.get(hook_type, [])

        for hook in hooks:
            result = hook.execute(current_context)
            results.append(result)

            # If hook modified context, use updated version
            if result.modified_context:
                current_context = result.modified_context

            # If hook failed critically, abort
            if result.abort:
                return (False, results)

        return (True, results)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all hooks"""
        stats = {}

        for hook_type, hooks in self.hooks.items():
            stats[hook_type.value] = [hook.get_stats() for hook in hooks]

        return stats


def create_default_hooks(heterogeneous_executor=None) -> HookManager:
    """
    Create hook manager with default hooks

    Args:
        heterogeneous_executor: Optional heterogeneous executor for backend selection

    Returns:
        Configured hook manager
    """
    manager = HookManager()

    # Pre-query hooks
    manager.register_hook(InputValidationHook())
    manager.register_hook(BackendSelectionHook(heterogeneous_executor))

    # Post-query hooks
    manager.register_hook(PerformanceLoggingHook())
    manager.register_hook(AutoOptimizationHook())

    # Git hooks
    manager.register_hook(PreCommitHook())

    return manager


def demo():
    """Demonstration of hook system"""
    print("=" * 70)
    print(" Hook System Demo")
    print("=" * 70)
    print()

    # Create hook manager
    manager = create_default_hooks()

    print()
    print("Testing pre-query hooks...")
    print()

    # Test input validation
    context = HookContext(
        hook_type=HookType.PRE_QUERY,
        timestamp=time.time(),
        data={"prompt": "Analyze this code"},
        metadata={}
    )

    success, results = manager.execute_hooks(HookType.PRE_QUERY, context)

    print(f"Pre-query hooks: {'✓ Success' if success else '❌ Failed'}")
    for result in results:
        print(f"  {result.message}")

    print()
    print("Testing post-query hooks...")
    print()

    # Test post-query hooks
    context = HookContext(
        hook_type=HookType.POST_QUERY,
        timestamp=time.time(),
        data={
            "prompt": "Test query",
            "latency_ms": 1500,
            "backend": "CPU",
            "success": True,
            "model": "test_model"
        },
        metadata={}
    )

    success, results = manager.execute_hooks(HookType.POST_QUERY, context)

    print(f"Post-query hooks: {'✓ Success' if success else '❌ Failed'}")
    for result in results:
        print(f"  {result.message}")

    print()
    print("=" * 70)
    print(" Hook Statistics")
    print("=" * 70)
    print()

    stats = manager.get_all_stats()
    for hook_type, hook_stats in stats.items():
        print(f"{hook_type}:")
        for stat in hook_stats:
            print(f"  {stat['name']}: {stat['executions']} executions, {stat['avg_time_ms']:.2f}ms avg")
        print()


if __name__ == "__main__":
    demo()

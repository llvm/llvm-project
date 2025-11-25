#!/usr/bin/env python3
"""
Heretic Hook Integration - DSMIL TEMPEST Plugin

Integrates Heretic abliteration into the hook system for seamless
model safety configuration via DSMIL TEMPEST web interface.

Usage:
    from heretic_hook import register_heretic_hooks
    from hook_system import HookManager

    manager = HookManager()
    register_heretic_hooks(manager)
"""

import time
from typing import Dict, Any
from pathlib import Path

try:
    from hook_system import Hook, HookType, HookPriority, HookContext, HookResult
    from heretic_config import ConfigLoader, HereticSettings
    from heretic_evaluator import RefusalDetector
    HOOK_SYSTEM_AVAILABLE = True
except ImportError:
    HOOK_SYSTEM_AVAILABLE = False

try:
    from enhanced_ai_engine import EnhancedAIEngine, HERETIC_AVAILABLE
except ImportError:
    HERETIC_AVAILABLE = False


class HereticConfigHook(Hook):
    """
    Hook for Heretic configuration management.

    Loads Heretic configuration and makes it available to other hooks.
    """

    def __init__(self):
        super().__init__(
            name="heretic_config",
            hook_type=HookType.PRE_QUERY,
            priority=HookPriority.CRITICAL
        )
        self.config_path = Path(__file__).parent / "heretic_config.toml"
        self.settings = None
        self._load_config()

    def _load_config(self):
        """Load Heretic configuration"""
        if not HERETIC_AVAILABLE:
            return

        try:
            self.settings = ConfigLoader.load(
                config_file=self.config_path if self.config_path.exists() else None
            )
        except Exception as e:
            print(f"Warning: Failed to load Heretic config: {e}")

    def execute(self, context: HookContext) -> HookResult:
        """Add Heretic configuration to context"""
        if not self.settings:
            return HookResult(
                success=True,
                message="Heretic config not loaded"
            )

        # Add config to context metadata
        if "heretic" not in context.metadata:
            context.metadata["heretic"] = {}

        context.metadata["heretic"]["config"] = {
            "n_trials": self.settings.n_trials,
            "kl_divergence_scale": self.settings.kl_divergence_scale,
            "enabled": True
        }

        return HookResult(
            success=True,
            message="Heretic config loaded",
            modified_context=context
        )


class HereticRefusalDetectionHook(Hook):
    """
    Hook for detecting refusals in model responses.

    Analyzes responses and flags potential refusals for monitoring.
    """

    def __init__(self):
        super().__init__(
            name="heretic_refusal_detection",
            hook_type=HookType.POST_QUERY,
            priority=HookPriority.HIGH
        )
        self.detector = None
        if HERETIC_AVAILABLE:
            try:
                self.detector = RefusalDetector()
            except Exception as e:
                print(f"Warning: Failed to initialize refusal detector: {e}")

    def execute(self, context: HookContext) -> HookResult:
        """Detect refusals in response"""
        if not self.detector:
            return HookResult(success=True, message="Detector not available")

        # Get response from context
        response = context.data.get("response", "")
        if not response:
            return HookResult(success=True, message="No response to analyze")

        # Detect refusal
        is_refusal = self.detector.is_refusal(response)

        if is_refusal:
            reason = self.detector.get_refusal_reason(response)

            # Add refusal info to metadata
            if "heretic" not in context.metadata:
                context.metadata["heretic"] = {}

            context.metadata["heretic"]["refusal_detected"] = True
            context.metadata["heretic"]["refusal_reason"] = reason

            return HookResult(
                success=True,
                message=f"Refusal detected: {reason}",
                data={"refusal": True, "reason": reason},
                modified_context=context
            )

        return HookResult(
            success=True,
            message="No refusal detected",
            data={"refusal": False}
        )


class HereticSafetyMonitorHook(Hook):
    """
    Hook for monitoring model safety metrics over time.

    Tracks refusal rates and alerts when safety constraints are active.
    """

    def __init__(self):
        super().__init__(
            name="heretic_safety_monitor",
            hook_type=HookType.PERFORMANCE_MONITOR,
            priority=HookPriority.NORMAL
        )
        self.refusal_count = 0
        self.total_queries = 0
        self.refusal_history = []

    def execute(self, context: HookContext) -> HookResult:
        """Monitor safety metrics"""
        self.total_queries += 1

        # Check if refusal was detected
        refusal_detected = (
            context.metadata.get("heretic", {}).get("refusal_detected", False)
        )

        if refusal_detected:
            self.refusal_count += 1
            self.refusal_history.append({
                "timestamp": context.timestamp,
                "reason": context.metadata["heretic"].get("refusal_reason")
            })

        # Calculate refusal rate
        refusal_rate = self.refusal_count / max(self.total_queries, 1)

        # Add metrics to context
        if "heretic" not in context.metadata:
            context.metadata["heretic"] = {}

        context.metadata["heretic"]["safety_metrics"] = {
            "refusal_count": self.refusal_count,
            "total_queries": self.total_queries,
            "refusal_rate": refusal_rate,
            "recent_refusals": len([
                r for r in self.refusal_history
                if context.timestamp - r["timestamp"] < 3600  # Last hour
            ])
        }

        return HookResult(
            success=True,
            message=f"Safety metrics updated: {refusal_rate:.1%} refusal rate",
            data={
                "refusal_rate": refusal_rate,
                "refusal_count": self.refusal_count
            },
            modified_context=context
        )


class HereticAbliterationTriggerHook(Hook):
    """
    Hook that can trigger automatic abliteration based on refusal rate.

    If refusal rate exceeds threshold, suggests or triggers abliteration.
    """

    def __init__(self, refusal_threshold: float = 0.5, auto_abliterate: bool = False):
        super().__init__(
            name="heretic_abliteration_trigger",
            hook_type=HookType.OPTIMIZATION,
            priority=HookPriority.HIGH
        )
        self.refusal_threshold = refusal_threshold
        self.auto_abliterate = auto_abliterate
        self.abliteration_triggered = False

    def execute(self, context: HookContext) -> HookResult:
        """Check if abliteration should be triggered"""
        safety_metrics = context.metadata.get("heretic", {}).get("safety_metrics", {})
        refusal_rate = safety_metrics.get("refusal_rate", 0.0)

        if refusal_rate > self.refusal_threshold and not self.abliteration_triggered:
            message = (
                f"High refusal rate detected: {refusal_rate:.1%} "
                f"(threshold: {self.refusal_threshold:.1%})"
            )

            if self.auto_abliterate:
                # Trigger automatic abliteration
                self.abliteration_triggered = True

                return HookResult(
                    success=True,
                    message=f"{message} - Triggering automatic abliteration",
                    data={
                        "action": "abliterate",
                        "refusal_rate": refusal_rate
                    }
                )
            else:
                # Just suggest abliteration
                return HookResult(
                    success=True,
                    message=f"{message} - Abliteration recommended",
                    data={
                        "action": "recommend_abliteration",
                        "refusal_rate": refusal_rate
                    }
                )

        return HookResult(
            success=True,
            message=f"Refusal rate OK: {refusal_rate:.1%}"
        )


class HereticWebInterfaceHook(Hook):
    """
    Hook that provides data for DSMIL TEMPEST web interface.

    Collects Heretic statistics and status for dashboard display.
    """

    def __init__(self):
        super().__init__(
            name="heretic_web_interface",
            hook_type=HookType.PERFORMANCE_MONITOR,
            priority=HookPriority.LOW
        )
        self.last_update = 0
        self.cached_data = {}

    def execute(self, context: HookContext) -> HookResult:
        """Collect data for web interface"""
        current_time = time.time()

        # Update cache every 5 seconds
        if current_time - self.last_update < 5.0:
            return HookResult(
                success=True,
                message="Using cached data",
                data=self.cached_data
            )

        # Collect fresh data
        heretic_data = context.metadata.get("heretic", {})

        self.cached_data = {
            "available": HERETIC_AVAILABLE,
            "refusal_detection": {
                "enabled": "refusal_detected" in heretic_data,
                "last_refusal": heretic_data.get("refusal_reason")
            },
            "safety_metrics": heretic_data.get("safety_metrics", {}),
            "config": heretic_data.get("config", {}),
            "timestamp": current_time
        }

        self.last_update = current_time

        return HookResult(
            success=True,
            message="Web interface data updated",
            data=self.cached_data
        )


# ===== REGISTRATION FUNCTION =====

def register_heretic_hooks(
    hook_manager,
    enable_refusal_detection: bool = True,
    enable_safety_monitor: bool = True,
    enable_abliteration_trigger: bool = False,
    auto_abliterate: bool = False,
    refusal_threshold: float = 0.5
):
    """
    Register all Heretic hooks with the hook manager.

    Args:
        hook_manager: HookManager instance
        enable_refusal_detection: Enable refusal detection hook
        enable_safety_monitor: Enable safety monitoring hook
        enable_abliteration_trigger: Enable abliteration trigger hook
        auto_abliterate: Auto-trigger abliteration (requires trigger enabled)
        refusal_threshold: Refusal rate threshold for triggering

    Returns:
        List of registered hook names
    """
    if not HOOK_SYSTEM_AVAILABLE:
        print("Warning: Hook system not available")
        return []

    registered = []

    # Always register config hook
    config_hook = HereticConfigHook()
    hook_manager.register_hook(config_hook)
    registered.append(config_hook.name)
    print(f"✓ Registered hook: {config_hook.name}")

    # Refusal detection hook
    if enable_refusal_detection:
        refusal_hook = HereticRefusalDetectionHook()
        hook_manager.register_hook(refusal_hook)
        registered.append(refusal_hook.name)
        print(f"✓ Registered hook: {refusal_hook.name}")

    # Safety monitor hook
    if enable_safety_monitor:
        safety_hook = HereticSafetyMonitorHook()
        hook_manager.register_hook(safety_hook)
        registered.append(safety_hook.name)
        print(f"✓ Registered hook: {safety_hook.name}")

    # Abliteration trigger hook
    if enable_abliteration_trigger:
        trigger_hook = HereticAbliterationTriggerHook(
            refusal_threshold=refusal_threshold,
            auto_abliterate=auto_abliterate
        )
        hook_manager.register_hook(trigger_hook)
        registered.append(trigger_hook.name)
        print(f"✓ Registered hook: {trigger_hook.name}")

    # Web interface hook (always enabled)
    web_hook = HereticWebInterfaceHook()
    hook_manager.register_hook(web_hook)
    registered.append(web_hook.name)
    print(f"✓ Registered hook: {web_hook.name}")

    print(f"\n🔬 Heretic: Registered {len(registered)} hooks")

    return registered


# ===== STANDALONE TESTING =====

if __name__ == "__main__":
    print("Heretic Hook System - Integration Test")
    print("=" * 60)

    if not HOOK_SYSTEM_AVAILABLE:
        print("❌ Hook system not available")
    elif not HERETIC_AVAILABLE:
        print("❌ Heretic modules not available")
    else:
        from hook_system import HookManager

        # Create hook manager
        manager = HookManager()

        # Register Heretic hooks
        hooks = register_heretic_hooks(
            manager,
            enable_refusal_detection=True,
            enable_safety_monitor=True,
            enable_abliteration_trigger=True,
            auto_abliterate=False,
            refusal_threshold=0.3
        )

        print(f"\n✅ Successfully registered {len(hooks)} Heretic hooks")
        print("\nRegistered hooks:")
        for hook_name in hooks:
            print(f"  - {hook_name}")

        # Test hook execution
        print("\n" + "=" * 60)
        print("Testing hook execution...")

        test_context = HookContext(
            hook_type=HookType.POST_QUERY,
            timestamp=time.time(),
            data={
                "response": "I'm sorry, I can't help with that request."
            },
            metadata={}
        )

        # Execute hooks
        results = manager.execute_hooks(HookType.POST_QUERY, test_context)

        print(f"\nExecuted {len(results)} hooks:")
        for result in results:
            print(f"  - {result.message}")

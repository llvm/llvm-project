#!/usr/bin/env python3
"""
DSMIL Real-Time Safety Monitoring System

Provides continuous real-time monitoring of system health, device states,
and safety metrics during DSMIL operations. Implements automatic emergency
stop on critical conditions.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import signal
import argparse
from typing import Dict, List
from datetime import datetime

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dsmil_safety import SafetyValidator, SAFE_DEVICES
from dsmil_common import DeviceAccess
from dsmil_logger import create_logger, LogLevel

class SafetyMonitor:
    """Real-time safety monitoring system"""

    def __init__(self, logger, safety, interval=1.0):
        self.logger = logger
        self.safety = safety
        self.interval = interval
        self.running = False
        self.metrics_history = []
        self.alert_count = 0
        self.emergency_stops = 0

    def start(self):
        """Start monitoring loop"""
        self.running = True
        self.logger.info("monitor", "Safety monitoring started")

        print("=" * 80)
        print("DSMIL Real-Time Safety Monitor")
        print("=" * 80)
        print("Monitoring system health and device states...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Check system health
                healthy, metrics = self.safety.check_system_health()

                # Log metrics
                self.metrics_history.append({
                    "timestamp": time.time(),
                    "metrics": metrics,
                    "healthy": healthy,
                })

                # Keep only last 100 entries
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

                # Display current status
                self._display_status(metrics, healthy)

                # Check for alerts
                if not healthy:
                    self.alert_count += 1
                    self.logger.warning("monitor", "System health check failed", data=metrics)

                    # Check if emergency stop needed
                    if self._check_emergency_conditions(metrics):
                        self.emergency_stops += 1
                        self.safety.emergency_stop("Critical system condition detected")
                        self.logger.critical("monitor", "EMERGENCY STOP TRIGGERED", data=metrics)
                        break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.logger.info("monitor", "Monitoring stopped by user")

        finally:
            self.running = False
            self._print_summary()

    def _display_status(self, metrics: Dict, healthy: bool):
        """Display current status"""
        status_char = "✓" if healthy else "✗"
        health_str = "HEALTHY" if healthy else "ALERT"

        # Get timestamp
        ts = datetime.now().strftime("%H:%M:%S")

        # Format metrics
        uptime_hrs = metrics.get("uptime", 0) / 3600
        load = metrics.get("load_average", 0)
        mem_mb = metrics.get("memory_available", 0) / (1024 * 1024)
        disk_gb = metrics.get("disk_space", 0) / (1024 * 1024 * 1024)
        thermal = "OK" if metrics.get("thermal_ok") else "HIGH"

        print(f"[{ts}] {status_char} {health_str:8} | "
              f"Uptime: {uptime_hrs:6.1f}h | "
              f"Load: {load:4.1f} | "
              f"Mem: {mem_mb:6.0f}MB | "
              f"Disk: {disk_gb:5.1f}GB | "
              f"Thermal: {thermal}", end='\r')

    def _check_emergency_conditions(self, metrics: Dict) -> bool:
        """Check if emergency stop is needed"""
        # Critical conditions
        if metrics.get("load_average", 0) > 20:
            return True
        if metrics.get("memory_available", float('inf')) < 50 * 1024 * 1024:  # < 50MB
            return True
        if metrics.get("disk_space", float('inf')) < 100 * 1024 * 1024:  # < 100MB
            return True
        if not metrics.get("thermal_ok"):
            return True

        return False

    def _print_summary(self):
        """Print monitoring summary"""
        print("\n\n" + "=" * 80)
        print("Monitoring Session Summary")
        print("=" * 80)
        print(f"Total samples: {len(self.metrics_history)}")
        print(f"Alerts: {self.alert_count}")
        print(f"Emergency stops: {self.emergency_stops}")

        if self.metrics_history:
            # Calculate averages
            avg_load = sum(m["metrics"].get("load_average", 0) for m in self.metrics_history) / len(self.metrics_history)
            avg_mem = sum(m["metrics"].get("memory_available", 0) for m in self.metrics_history) / len(self.metrics_history)

            print(f"\nAverage Metrics:")
            print(f"  Load Average: {avg_load:.2f}")
            print(f"  Memory Available: {avg_mem / (1024 * 1024):.0f} MB")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Real-Time Safety Monitoring System"
    )

    parser.add_argument('--interval', type=float, default=1.0,
                       help='Monitoring interval in seconds (default: 1.0)')
    parser.add_argument('--log-dir', default='output/monitor_logs',
                       help='Log output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Create logger
    log_level = LogLevel.DEBUG if args.verbose else LogLevel.INFO
    logger = create_logger(log_dir=args.log_dir, min_level=log_level)

    # Create safety validator
    safety = SafetyValidator()

    # Create monitor
    monitor = SafetyMonitor(logger, safety, interval=args.interval)

    try:
        monitor.start()
    except Exception as e:
        print(f"\nError: {e}")
        logger.error("monitor", f"Fatal error: {e}")
        return 1
    finally:
        logger.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())

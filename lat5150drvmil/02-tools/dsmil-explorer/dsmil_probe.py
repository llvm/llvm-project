#!/usr/bin/env python3
"""
DSMIL Automated Safe Probing Framework

Implements the 5-phase safe exploration methodology for unknown DSMIL devices.
Provides progressive device interrogation with comprehensive safety checks.

Phases:
  1. RECONNAISSANCE - Passive capability reading
  2. PASSIVE OBSERVATION - Read-only monitoring
  3. CONTROLLED TESTING - Isolated single operations
  4. INCREMENTAL ENABLING - Supervised feature activation
  5. PRODUCTION INTEGRATION - Full validation and deployment

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dsmil_safety import SafetyValidator, SafetyLevel, DeviceRisk, QUARANTINED_DEVICES
from dsmil_common import DSMILDevice, DeviceAccess, check_kernel_module_loaded
from dsmil_logger import DSMILLogger, LogLevel, create_logger

class ProbePhase(Enum):
    """Probe phases for progressive exploration"""
    RECONNAISSANCE = 1
    PASSIVE_OBSERVATION = 2
    CONTROLLED_TESTING = 3
    INCREMENTAL_ENABLING = 4
    PRODUCTION_INTEGRATION = 5

class ProbeResult:
    """Results from device probing"""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.timestamp = time.time()
        self.phases_completed = []
        self.capabilities = {}
        self.observations = {}
        self.test_results = {}
        self.safety_checks = {}
        self.errors = []
        self.warnings = []
        self.success = False

    def add_phase_result(self, phase: ProbePhase, data: Dict):
        """Add results from a phase"""
        self.phases_completed.append(phase.name)

        if phase == ProbePhase.RECONNAISSANCE:
            self.capabilities = data
        elif phase == ProbePhase.PASSIVE_OBSERVATION:
            self.observations = data
        elif phase == ProbePhase.CONTROLLED_TESTING:
            self.test_results = data

    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "device_id": f"0x{self.device_id:04X}",
            "timestamp": self.timestamp,
            "phases_completed": self.phases_completed,
            "capabilities": self.capabilities,
            "observations": self.observations,
            "test_results": self.test_results,
            "safety_checks": self.safety_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
        }

class AutomatedProber:
    """Automated device probing with 5-phase methodology"""

    def __init__(self, logger: DSMILLogger, safety: SafetyValidator,
                 dry_run: bool = False):
        self.logger = logger
        self.safety = safety
        self.dry_run = dry_run
        self.results = {}

    def probe_device(self, device_id: int, max_phase: ProbePhase = ProbePhase.PASSIVE_OBSERVATION) -> ProbeResult:
        """
        Probe a single device through multiple phases

        Args:
            device_id: Device to probe
            max_phase: Maximum phase to execute (default: PASSIVE_OBSERVATION)

        Returns:
            ProbeResult with all collected data
        """
        result = ProbeResult(device_id)

        self.logger.info("probe", f"Starting probe sequence for device 0x{device_id:04X}",
                        device_id=device_id)

        # Safety check first
        allowed, reason, level = self.safety.check_device_access(device_id)

        if not allowed:
            self.logger.error("probe", f"Device access forbidden: {reason}",
                             device_id=device_id)
            result.add_error(f"Access forbidden: {reason}")
            return result

        if level == SafetyLevel.FORBIDDEN:
            self.logger.critical("probe", "Attempted access to quarantined device!",
                                device_id=device_id)
            result.add_error("Device is quarantined - NEVER ACCESS")
            return result

        # System health check
        healthy, health = self.safety.check_system_health()
        if not healthy:
            self.logger.warning("probe", "System health check failed",
                               device_id=device_id, data=health)
            result.add_warning("System health check failed - proceeding with caution")

        # Execute phases
        try:
            # Phase 1: Reconnaissance
            if self._execute_phase1_reconnaissance(device_id, result):
                if max_phase.value <= ProbePhase.RECONNAISSANCE.value:
                    result.success = True
                    return result

            # Phase 2: Passive Observation
            if max_phase.value >= ProbePhase.PASSIVE_OBSERVATION.value:
                if self._execute_phase2_observation(device_id, result):
                    if max_phase.value <= ProbePhase.PASSIVE_OBSERVATION.value:
                        result.success = True
                        return result

            # Phase 3: Controlled Testing (only if explicitly requested)
            if max_phase.value >= ProbePhase.CONTROLLED_TESTING.value:
                if level == SafetyLevel.RISKY:
                    self.logger.warning("probe", "Skipping controlled testing for risky device",
                                       device_id=device_id)
                    result.add_warning("Controlled testing skipped (device is risky)")
                elif not self.dry_run:
                    self._execute_phase3_testing(device_id, result)

            result.success = len(result.errors) == 0

        except Exception as e:
            self.logger.error("probe", f"Probing failed with exception: {e}",
                             device_id=device_id)
            result.add_error(f"Exception: {str(e)}")

        return result

    def _execute_phase1_reconnaissance(self, device_id: int, result: ProbeResult) -> bool:
        """Phase 1: Reconnaissance - passive capability reading"""
        self.logger.info("probe", "Phase 1: RECONNAISSANCE",
                        device_id=device_id)

        start_time = time.time()
        phase_data = {}

        try:
            if self.dry_run:
                self.logger.info("probe", "DRY RUN: Simulating reconnaissance",
                                device_id=device_id)
                phase_data = {
                    "simulated": True,
                    "capabilities_register": "0x00000000",
                    "version": "unknown",
                }
            else:
                with DeviceAccess() as dev:
                    if not dev.is_open:
                        result.add_error("Failed to open device driver")
                        return False

                    # Read device capabilities
                    caps = dev.get_device_capabilities(device_id)
                    if caps:
                        phase_data["capabilities"] = caps
                        self.logger.info("probe", "Capabilities read successfully",
                                        device_id=device_id, data=caps)
                    else:
                        phase_data["capabilities"] = None
                        result.add_warning("Could not read capabilities")

                    # Read device status
                    status = dev.get_device_status(device_id)
                    if status:
                        phase_data["status"] = status
                        self.logger.info("probe", "Status read successfully",
                                        device_id=device_id, data=status)
                    else:
                        phase_data["status"] = None
                        result.add_warning("Could not read status")

            duration = time.time() - start_time
            self.logger.log_operation("phase1_reconnaissance", device_id,
                                     success=True, duration=duration)

            result.add_phase_result(ProbePhase.RECONNAISSANCE, phase_data)
            return True

        except Exception as e:
            self.logger.error("probe", f"Phase 1 failed: {e}", device_id=device_id)
            result.add_error(f"Phase 1 failed: {str(e)}")
            return False

    def _execute_phase2_observation(self, device_id: int, result: ProbeResult) -> bool:
        """Phase 2: Passive Observation - read-only monitoring"""
        self.logger.info("probe", "Phase 2: PASSIVE OBSERVATION",
                        device_id=device_id)

        start_time = time.time()
        phase_data = {}

        try:
            if self.dry_run:
                self.logger.info("probe", "DRY RUN: Simulating observation",
                                device_id=device_id)
                phase_data = {
                    "simulated": True,
                    "register_values": ["0x00", "0x00", "0x00", "0x00"],
                    "observations": "No anomalies detected (simulated)",
                }
            else:
                with DeviceAccess() as dev:
                    if not dev.is_open:
                        result.add_error("Failed to open device driver")
                        return False

                    # Read first few registers (read-only, safe)
                    register_values = []
                    for offset in range(0, 16, 4):  # Read first 4 registers
                        value = dev.read_register(device_id, offset, size=4)
                        if value is not None:
                            register_values.append(f"0x{value:08X}")
                            self.logger.debug("probe", f"Register 0x{offset:02X} = 0x{value:08X}",
                                            device_id=device_id)
                        else:
                            register_values.append("ERROR")
                            result.add_warning(f"Could not read register 0x{offset:02X}")

                    phase_data["register_values"] = register_values

                    # Monitor for changes over time (sample 3 times, 1 second apart)
                    samples = []
                    for i in range(3):
                        if i > 0:
                            time.sleep(1)

                        sample = {
                            "time": time.time(),
                            "status": dev.get_device_status(device_id),
                        }
                        samples.append(sample)

                    phase_data["samples"] = samples

                    # Check for anomalies
                    anomalies = self._detect_anomalies(samples)
                    phase_data["anomalies"] = anomalies

                    if anomalies:
                        self.logger.warning("probe", f"Detected {len(anomalies)} anomalies",
                                          device_id=device_id, data={"anomalies": anomalies})
                        result.add_warning(f"Detected {len(anomalies)} anomalies")

            duration = time.time() - start_time
            self.logger.log_operation("phase2_observation", device_id,
                                     success=True, duration=duration)

            result.add_phase_result(ProbePhase.PASSIVE_OBSERVATION, phase_data)
            return True

        except Exception as e:
            self.logger.error("probe", f"Phase 2 failed: {e}", device_id=device_id)
            result.add_error(f"Phase 2 failed: {str(e)}")
            return False

    def _execute_phase3_testing(self, device_id: int, result: ProbeResult) -> bool:
        """Phase 3: Controlled Testing - isolated single operations"""
        self.logger.warning("probe", "Phase 3: CONTROLLED TESTING (high risk)",
                           device_id=device_id)

        # Phase 3 is high-risk and requires manual approval
        result.add_warning("Phase 3 requires manual approval and is not automated")
        return False

    def _detect_anomalies(self, samples: List[Dict]) -> List[str]:
        """Detect anomalies in observation samples"""
        anomalies = []

        # Check if status changed unexpectedly
        if len(samples) >= 2:
            for i in range(1, len(samples)):
                prev_status = samples[i-1].get("status", {})
                curr_status = samples[i].get("status", {})

                if prev_status and curr_status:
                    # Check if error flag was set
                    if not prev_status.get("error") and curr_status.get("error"):
                        anomalies.append("Error flag set during observation")

                    # Check if device became disabled
                    if prev_status.get("enabled") and not curr_status.get("enabled"):
                        anomalies.append("Device became disabled during observation")

        return anomalies

    def probe_device_range(self, start_id: int, end_id: int,
                          max_phase: ProbePhase = ProbePhase.PASSIVE_OBSERVATION) -> Dict[int, ProbeResult]:
        """
        Probe a range of devices

        Args:
            start_id: Starting device ID
            end_id: Ending device ID (inclusive)
            max_phase: Maximum phase to execute

        Returns:
            Dictionary of device_id -> ProbeResult
        """
        self.logger.info("probe", f"Starting range probe: 0x{start_id:04X} to 0x{end_id:04X}")

        results = {}

        for device_id in range(start_id, end_id + 1):
            # Skip quarantined devices
            if device_id in QUARANTINED_DEVICES:
                self.logger.warning("probe", f"Skipping quarantined device 0x{device_id:04X}",
                                   device_id=device_id)
                continue

            # Probe device
            result = self.probe_device(device_id, max_phase)
            results[device_id] = result

            # Small delay between devices
            time.sleep(0.5)

        self.logger.info("probe", f"Range probe complete: {len(results)} devices probed")

        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Automated Safe Probing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe single device (reconnaissance only)
  sudo python3 dsmil_probe.py --device 0x8030

  # Probe device with full observation
  sudo python3 dsmil_probe.py --device 0x8030 --phase 2

  # Probe range of devices (Group 3)
  sudo python3 dsmil_probe.py --range 0x8030 0x803B

  # Dry run (no actual hardware access)
  sudo python3 dsmil_probe.py --device 0x8030 --dry-run

Safety Levels:
  Phase 1 (RECONNAISSANCE): Read capabilities only (safest)
  Phase 2 (PASSIVE_OBSERVATION): Read-only monitoring (safe)
  Phase 3 (CONTROLLED_TESTING): Single operations (requires manual approval)
        """
    )

    parser.add_argument('--device', type=lambda x: int(x, 0), metavar='0xXXXX',
                       help='Single device ID to probe (hex format)')
    parser.add_argument('--range', nargs=2, type=lambda x: int(x, 0), metavar=('START', 'END'),
                       help='Range of devices to probe (hex format)')
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2],
                       help='Maximum probe phase (1=reconnaissance, 2=observation)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (no actual hardware access)')
    parser.add_argument('--log-dir', default='output/probe_logs',
                       help='Log output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Validate arguments
    if not args.device and not args.range:
        parser.error("Must specify either --device or --range")

    # Check kernel module
    if not args.dry_run and not check_kernel_module_loaded():
        print("Error: DSMIL kernel module not loaded")
        print("Load with: sudo insmod 01-source/kernel/dsmil-72dev.ko")
        return 1

    # Create logger
    log_level = LogLevel.DEBUG if args.verbose else LogLevel.INFO
    logger = create_logger(log_dir=args.log_dir, min_level=log_level)

    # Create safety validator
    safety = SafetyValidator()

    # Create prober
    prober = AutomatedProber(logger, safety, dry_run=args.dry_run)

    # Map phase number to enum
    max_phase = ProbePhase.RECONNAISSANCE if args.phase == 1 else ProbePhase.PASSIVE_OBSERVATION

    try:
        if args.device:
            # Probe single device
            result = prober.probe_device(args.device, max_phase)

            print("\n" + "=" * 80)
            print(f"Probe Results for Device 0x{args.device:04X}")
            print("=" * 80)
            print(f"Success: {result.success}")
            print(f"Phases completed: {', '.join(result.phases_completed)}")
            print(f"Errors: {len(result.errors)}")
            print(f"Warnings: {len(result.warnings)}")

            if result.capabilities:
                print("\nCapabilities:")
                for key, value in result.capabilities.items():
                    print(f"  {key}: {value}")

        else:
            # Probe range
            start_id, end_id = args.range
            results = prober.probe_device_range(start_id, end_id, max_phase)

            print("\n" + "=" * 80)
            print(f"Probe Results for Range 0x{start_id:04X} - 0x{end_id:04X}")
            print("=" * 80)

            successful = sum(1 for r in results.values() if r.success)
            print(f"Devices probed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")

            # Summary per device
            print("\nPer-Device Summary:")
            for device_id, result in sorted(results.items()):
                status = "✓" if result.success else "✗"
                print(f"  0x{device_id:04X}: {status} [{len(result.phases_completed)} phases, {len(result.errors)} errors]")

    except KeyboardInterrupt:
        print("\n\nProbing interrupted by user")
        logger.warning("probe", "Probing interrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        logger.error("probe", f"Fatal error: {e}")
        return 1

    finally:
        # Show statistics
        stats = logger.get_statistics()
        print("\n" + "=" * 80)
        print("Session Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 80)

        logger.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())

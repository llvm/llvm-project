#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: DSMILIntegrationAdapter.activate_device()
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Device Activation Implementation
Comprehensive device activation with safety checks and rollback

Supports 3 activation methods:
1. Kernel ioctl interface (/dev/dsmil)
2. sysfs interface (/sys/devices/platform/dell-milspec/)
3. Direct SMI calls (I/O ports 0x164E/0x164F with iopl(3))
"""

import os
import sys
import subprocess
import struct
import fcntl
import time
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsmil_subsystem_controller import DSMILSubsystemController, SubsystemType
from dsmil_device_database import QUARANTINED_DEVICES, SAFE_DEVICES, DEVICE_DATABASE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_device_activation.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  from dsmil_integration_adapter import DSMILIntegrationAdapter\n"
    "  adapter = DSMILIntegrationAdapter()\n"
    "  success = adapter.activate_device(device_id)\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class ActivationMethod(Enum):
    """Device activation method"""
    IOCTL = "ioctl"  # Kernel ioctl interface
    SYSFS = "sysfs"  # sysfs interface
    SMI = "smi"      # Direct SMI calls (requires iopl(3))


class ActivationStatus(Enum):
    """Activation status"""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ActivationResult:
    """Result of device activation"""
    device_id: int
    device_name: str
    status: ActivationStatus
    method: Optional[ActivationMethod]
    success: bool
    message: str
    timestamp: str
    rollback_available: bool
    thermal_impact: Optional[float] = None
    power_delta: Optional[float] = None


class DSMILDeviceActivator:
    """Comprehensive DSMIL device activation with safety checks"""

    # IOCTL commands (from dell-milspec.h)
    MILSPEC_IOC_MAGIC = ord('M')
    MILSPEC_IOC_ACTIVATE_DSMIL = (2 << 30) | (MILSPEC_IOC_MAGIC << 8) | (3 << 0) | (4 << 16)

    def __init__(self, sudo_password: Optional[str] = None):
        self.controller = DSMILSubsystemController()
        self.sudo_password = sudo_password
        self.activation_history: List[ActivationResult] = []
        self.rollback_points: Dict[int, Dict] = {}

        # Detect available activation methods
        self.available_methods = self._detect_activation_methods()
        logger.info(f"Available activation methods: {[m.value for m in self.available_methods]}")

    def _detect_activation_methods(self) -> List[ActivationMethod]:
        """Detect which activation methods are available"""
        methods = []

        # Check for kernel device
        if Path("/dev/dsmil").exists():
            methods.append(ActivationMethod.IOCTL)
            logger.info("✓ Kernel ioctl interface available")

        # Check for sysfs interface
        if Path("/sys/devices/platform/dell-milspec").exists():
            methods.append(ActivationMethod.SYSFS)
            logger.info("✓ sysfs interface available")

        # SMI is always theoretically available but requires root
        if os.geteuid() == 0:
            methods.append(ActivationMethod.SMI)
            logger.info("✓ Direct SMI access available (root)")
        else:
            logger.warning("✗ Direct SMI requires root privileges")

        return methods

    def _run_sudo_command(self, cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
        """Execute command with sudo"""
        try:
            if self.sudo_password and os.geteuid() != 0:
                full_cmd = ["sudo", "-S"] + cmd
                process = subprocess.Popen(
                    full_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(
                    input=self.sudo_password + "\n",
                    timeout=timeout
                )
                return process.returncode, stdout, stderr
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return -1, "", "Timeout"
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return -1, "", str(e)

    def validate_device_safety(self, device_id: int) -> Tuple[bool, str]:
        """Comprehensive device safety validation"""

        # Check 1: Quarantine list
        if device_id in QUARANTINED_DEVICES:
            return False, f"CRITICAL: Device 0x{device_id:04X} is QUARANTINED - activation absolutely forbidden"

        # Check 2: Device exists in database
        if device_id not in DEVICE_DATABASE:
            return False, f"Device 0x{device_id:04X} not found in device database"

        # Check 3: Device control subsystem operational
        device_control = self.controller.get_subsystem_status(SubsystemType.DEVICE_CONTROL)
        if not device_control or not device_control.operational:
            return False, "Device control subsystem not operational"

        # Check 4: Thermal conditions
        thermal_status = self.controller.get_thermal_status_enhanced()
        max_temp = thermal_status.get('max_temp', 0)
        if max_temp > 90:
            return False, f"Thermal critical: {max_temp}°C exceeds safe activation threshold (90°C)"
        elif max_temp > 85:
            logger.warning(f"Thermal warning: {max_temp}°C approaching limit")

        # Check 5: TPM attestation for secure devices
        device_info = DEVICE_DATABASE[device_id]
        if device_info.get('security_critical', False):
            tpm_status = self.controller.get_subsystem_status(SubsystemType.TPM_ATTESTATION)
            if not tpm_status or not tpm_status.operational:
                return False, "TPM attestation required for security-critical device but unavailable"

        return True, "Device validated safe for activation"

    def create_rollback_point(self, device_id: int):
        """Create rollback point before activation"""
        logger.info(f"Creating rollback point for device 0x{device_id:04X}")

        device_status = self.controller.get_device_status_cached(device_id)
        thermal_status = self.controller.get_thermal_status_enhanced()

        self.rollback_points[device_id] = {
            'timestamp': datetime.now().isoformat(),
            'device_status': device_status,
            'thermal_state': thermal_status,
            'operation_count': len(self.controller.get_operation_history(limit=1))
        }

        logger.info(f"✓ Rollback point created for 0x{device_id:04X}")

    def activate_via_ioctl(self, device_id: int, value: int = 1) -> Tuple[bool, str]:
        """Activate device via kernel ioctl interface"""
        device_path = Path("/dev/dsmil")

        if not device_path.exists():
            return False, "Kernel device /dev/dsmil not available"

        try:
            with open(device_path, 'r+b') as f:
                # Pack ioctl data: device_id as u32
                data = struct.pack('I', device_id)

                # Call ioctl
                result = fcntl.ioctl(f, self.MILSPEC_IOC_ACTIVATE_DSMIL, data)

                logger.info(f"✓ Device 0x{device_id:04X} activated via ioctl (result: {result})")
                return True, f"Activated via ioctl (result: {result})"

        except OSError as e:
            logger.error(f"ioctl failed: {e}")
            return False, f"ioctl error: {e}"
        except Exception as e:
            logger.error(f"Activation failed: {e}")
            return False, str(e)

    def activate_via_sysfs(self, device_id: int, value: int = 1) -> Tuple[bool, str]:
        """Activate device via sysfs interface"""
        sysfs_base = Path("/sys/devices/platform/dell-milspec")

        if not sysfs_base.exists():
            return False, "sysfs interface not available"

        # Look for device-specific sysfs entry
        device_sysfs = sysfs_base / f"device_{device_id:04X}" / "activate"

        if not device_sysfs.exists():
            # Try alternative path
            device_sysfs = sysfs_base / "devices" / f"{device_id:04X}" / "activate"

        if not device_sysfs.exists():
            return False, f"sysfs entry not found for device 0x{device_id:04X}"

        try:
            # Write activation value to sysfs
            device_sysfs.write_text(str(value))
            logger.info(f"✓ Device 0x{device_id:04X} activated via sysfs")
            return True, "Activated via sysfs"
        except PermissionError:
            return False, "Permission denied (need root)"
        except Exception as e:
            logger.error(f"sysfs activation failed: {e}")
            return False, str(e)

    def activate_via_smi(self, device_id: int, value: int = 1) -> Tuple[bool, str]:
        """Activate device via direct SMI calls (requires iopl(3))"""
        if os.geteuid() != 0:
            return False, "SMI activation requires root privileges"

        try:
            # This would require ctypes and iopl(3) for direct I/O port access
            # For now, simulate the operation
            logger.info(f"SMI activation for device 0x{device_id:04X} would write to ports 0x164E/0x164F")

            # Actual implementation would be:
            # import ctypes
            # libc = ctypes.CDLL(None)
            # libc.iopl(3)  # Request I/O privilege level 3
            # outw(device_id, 0x164E)  # Write device token
            # outb(value, 0x164F)  # Write activation value
            # status = inb(0x164F)  # Read status

            return False, "SMI activation requires kernel module with proper I/O access"
        except Exception as e:
            logger.error(f"SMI activation failed: {e}")
            return False, str(e)

    def activate_device(
        self,
        device_id: int,
        value: int = 1,
        preferred_method: Optional[ActivationMethod] = None
    ) -> ActivationResult:
        """Activate a DSMIL device with comprehensive safety checks"""

        device_info = DEVICE_DATABASE.get(device_id, {})
        device_name = device_info.get('name', f'UNKNOWN_0x{device_id:04X}')

        logger.info(f"{'='*60}")
        logger.info(f"Activating device 0x{device_id:04X}: {device_name}")
        logger.info(f"{'='*60}")

        # Safety validation
        safe, safety_message = self.validate_device_safety(device_id)
        if not safe:
            logger.error(f"✗ Safety validation failed: {safety_message}")
            result = ActivationResult(
                device_id=device_id,
                device_name=device_name,
                status=ActivationStatus.FAILED,
                method=None,
                success=False,
                message=safety_message,
                timestamp=datetime.now().isoformat(),
                rollback_available=False
            )
            self.activation_history.append(result)
            self.controller.log_operation(device_id, 'activate', False, safety_message)
            return result

        logger.info(f"✓ Safety validation passed: {safety_message}")

        # Create rollback point
        self.create_rollback_point(device_id)

        # Record thermal baseline
        thermal_before = self.controller.get_thermal_status_enhanced()
        temp_before = thermal_before.get('max_temp', 0)

        # Try activation methods in order of preference
        methods_to_try = []
        if preferred_method and preferred_method in self.available_methods:
            methods_to_try.append(preferred_method)
        methods_to_try.extend([m for m in self.available_methods if m != preferred_method])

        success = False
        method_used = None
        activation_message = ""

        for method in methods_to_try:
            logger.info(f"Attempting activation via {method.value}...")

            if method == ActivationMethod.IOCTL:
                success, activation_message = self.activate_via_ioctl(device_id, value)
            elif method == ActivationMethod.SYSFS:
                success, activation_message = self.activate_via_sysfs(device_id, value)
            elif method == ActivationMethod.SMI:
                success, activation_message = self.activate_via_smi(device_id, value)

            if success:
                method_used = method
                logger.info(f"✓ Activation successful via {method.value}")
                break
            else:
                logger.warning(f"✗ Activation via {method.value} failed: {activation_message}")

        # Verify activation and measure thermal impact
        time.sleep(0.5)  # Brief delay for device stabilization
        thermal_after = self.controller.get_thermal_status_enhanced()
        temp_after = thermal_after.get('max_temp', 0)
        thermal_impact = temp_after - temp_before

        # Create result
        final_status = ActivationStatus.ACTIVE if success else ActivationStatus.FAILED

        result = ActivationResult(
            device_id=device_id,
            device_name=device_name,
            status=final_status,
            method=method_used,
            success=success,
            message=activation_message if success else f"All activation methods failed. Last: {activation_message}",
            timestamp=datetime.now().isoformat(),
            rollback_available=device_id in self.rollback_points,
            thermal_impact=thermal_impact
        )

        # Log operation
        self.controller.log_operation(
            device_id,
            'activate',
            success,
            f"{method_used.value if method_used else 'failed'}: {activation_message}"
        )

        # Add to history
        self.activation_history.append(result)

        # Log thermal impact
        if thermal_impact > 2.0:
            logger.warning(f"⚠ Significant thermal impact: +{thermal_impact:.1f}°C")

        logger.info(f"{'='*60}")
        logger.info(f"Activation {'SUCCEEDED' if success else 'FAILED'}")
        logger.info(f"{'='*60}\n")

        return result

    def rollback_device(self, device_id: int) -> bool:
        """Rollback device to previous state"""
        if device_id not in self.rollback_points:
            logger.error(f"No rollback point available for device 0x{device_id:04X}")
            return False

        logger.warning(f"Rolling back device 0x{device_id:04X}...")

        # Attempt deactivation via all available methods
        success = False
        for method in self.available_methods:
            if method == ActivationMethod.IOCTL:
                success, _ = self.activate_via_ioctl(device_id, value=0)
            elif method == ActivationMethod.SYSFS:
                success, _ = self.activate_via_sysfs(device_id, value=0)
            elif method == ActivationMethod.SMI:
                success, _ = self.activate_via_smi(device_id, value=0)

            if success:
                break

        if success:
            logger.info(f"✓ Device 0x{device_id:04X} rolled back successfully")
            self.controller.log_operation(device_id, 'rollback', True, "Device deactivated")
        else:
            logger.error(f"✗ Rollback failed for device 0x{device_id:04X}")
            self.controller.log_operation(device_id, 'rollback', False, "Deactivation failed")

        return success

    def activate_safe_devices(self) -> List[ActivationResult]:
        """Activate all safe devices"""
        logger.info(f"Activating {len(SAFE_DEVICES)} safe devices...")

        results = []
        for device_id in SAFE_DEVICES:
            result = self.activate_device(device_id)
            results.append(result)

            # Brief pause between activations to monitor thermal
            time.sleep(1.0)

            # Check thermal conditions after each activation
            thermal = self.controller.get_thermal_status_enhanced()
            if thermal.get('overall_status') == 'critical':
                logger.error("THERMAL CRITICAL - Stopping activations")
                break

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\nActivation Summary: {successful}/{len(results)} devices activated successfully")

        return results

    def generate_activation_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive activation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_attempts': len(self.activation_history),
            'successful': sum(1 for r in self.activation_history if r.success),
            'failed': sum(1 for r in self.activation_history if not r.success),
            'methods_used': {},
            'thermal_impact': {},
            'devices': []
        }

        # Count method usage
        for result in self.activation_history:
            if result.method:
                method_name = result.method.value
                report['methods_used'][method_name] = report['methods_used'].get(method_name, 0) + 1

        # Thermal impact summary
        thermal_impacts = [r.thermal_impact for r in self.activation_history if r.thermal_impact is not None]
        if thermal_impacts:
            report['thermal_impact'] = {
                'max_increase': max(thermal_impacts),
                'avg_increase': sum(thermal_impacts) / len(thermal_impacts),
                'total_increase': sum(thermal_impacts)
            }

        # Individual device results
        for result in self.activation_history:
            report['devices'].append({
                'device_id': f"0x{result.device_id:04X}",
                'device_name': result.device_name,
                'status': result.status.value,
                'method': result.method.value if result.method else None,
                'success': result.success,
                'message': result.message,
                'timestamp': result.timestamp,
                'thermal_impact': result.thermal_impact
            })

        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Activation report saved to {output_path}")

        return report


def main():
    """Main entry point for device activation"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL Device Activation")
    parser.add_argument('--device', type=str, help="Device ID to activate (hex, e.g., 0x8000)")
    parser.add_argument('--safe-devices', action='store_true', help="Activate all safe devices")
    parser.add_argument('--method', type=str, choices=['ioctl', 'sysfs', 'smi'],
                       help="Preferred activation method")
    parser.add_argument('--value', type=int, default=1, help="Activation value (default: 1)")
    parser.add_argument('--report', type=str, help="Save activation report to file")
    parser.add_argument('--sudo-password', type=str, help="Sudo password for privileged operations")

    args = parser.parse_args()

    # Initialize activator
    activator = DSMILDeviceActivator(sudo_password=args.sudo_password)

    # Determine activation method
    method = None
    if args.method:
        method = ActivationMethod(args.method)

    # Execute activation
    if args.safe_devices:
        results = activator.activate_safe_devices()
    elif args.device:
        device_id = int(args.device, 16) if args.device.startswith('0x') else int(args.device)
        result = activator.activate_device(device_id, value=args.value, preferred_method=method)
        results = [result]
    else:
        parser.print_help()
        sys.exit(1)

    # Generate report
    report_path = Path(args.report) if args.report else None
    report = activator.generate_activation_report(output_path=report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ACTIVATION SUMMARY")
    print("=" * 70)
    print(f"Total Attempts: {report['total_attempts']}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    if report['thermal_impact']:
        print(f"Max Thermal Impact: +{report['thermal_impact']['max_increase']:.1f}°C")
    print("=" * 70)

    # Exit code based on success rate
    success_rate = report['successful'] / report['total_attempts'] if report['total_attempts'] > 0 else 0
    sys.exit(0 if success_rate == 1.0 else 1)


if __name__ == "__main__":
    main()

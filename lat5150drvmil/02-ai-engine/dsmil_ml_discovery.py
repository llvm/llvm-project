#!/usr/bin/env python3
"""
DSMIL ML-Enhanced Hardware Discovery
====================================
Advanced hardware address discovery system that uses machine learning to:
- Identify unknown DSMIL devices dynamically
- Predict device capabilities based on hardware signatures
- Classify device safety levels
- Optimize activation sequences

This is a MISSION-CRITICAL component for smooth end-to-end activation workflow.

Author: LAT5150DRVMIL AI Platform
Classification: DSMIL ML Integration
"""

import os
import sys
import json
import logging
import subprocess
import re
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Configure logging - use user-specific log file to avoid permission issues
import os
_log_user = os.environ.get('SUDO_USER') or os.environ.get('USER') or 'dsmil'
_log_file = f'/tmp/dsmil_ml_discovery_{_log_user}.log'
logging.basicConfig(
    filename=_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeviceSafetyLevel(Enum):
    """ML-predicted device safety level"""
    SAFE = "safe"
    MONITORED = "monitored"
    CAUTION = "caution"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"


class HardwareInterface(Enum):
    """Hardware interface type"""
    SMBIOS = "smbios"
    ACPI = "acpi"
    IOCTL = "ioctl"
    SYSFS = "sysfs"
    PCI = "pci"


@dataclass
class HardwareDevice:
    """Discovered hardware device"""
    address: int
    name: str
    interface: HardwareInterface
    signature: str
    capabilities: List[str]
    safety_level: DeviceSafetyLevel
    ml_confidence: float
    dependencies: List[int]
    activation_priority: int
    thermal_estimate: float


class DSMILMLDiscovery:
    """ML-Enhanced Hardware Discovery System"""

    # Known device address ranges
    DSMIL_RANGES = [
        (0x8000, 0x806B, "Primary DSMIL Range"),
        (0x400, 0x47F, "Extended Token Range A"),
        (0x480, 0x4FF, "Extended Token Range B"),
        (0x500, 0x57F, "Extended Token Range C"),
        (0x1000, 0x15FF, "High Range A"),
        (0x4400, 0x4FFF, "High Range B"),
        (0xF100, 0xF1FF, "F-Series Range"),
    ]

    def __init__(self):
        """Initialize ML discovery system"""
        self.discovered_devices: Dict[int, HardwareDevice] = {}
        self.device_patterns: Dict[str, float] = {}
        self.scan_history: List[Dict] = []

        # Load ML model/patterns (simplified for this implementation)
        self._load_device_signatures()

        logger.info("DSMIL ML Discovery System initialized")

    def _load_device_signatures(self):
        """Load known device signatures for ML classification"""
        # Device signature patterns based on known DSMIL devices
        self.device_patterns = {
            'security': ['crypto', 'key', 'auth', 'tpm', 'secure', 'integrity'],
            'network': ['ethernet', 'wifi', 'bluetooth', 'vpn', 'firewall', 'dns'],
            'storage': ['disk', 'raid', 'nvme', 'sata', 'storage', 'filesystem'],
            'thermal': ['thermal', 'cooling', 'fan', 'temperature', 'heat'],
            'power': ['power', 'battery', 'voltage', 'current', 'wattage'],
            'emergency': ['wipe', 'emergency', 'kill', 'blackout', 'panic', 'hidden'],
        }

        # Safety classifications based on function keywords
        self.safety_patterns = {
            DeviceSafetyLevel.SAFE: ['monitor', 'status', 'read', 'info', 'display'],
            DeviceSafetyLevel.MONITORED: ['control', 'manage', 'configure', 'adjust'],
            DeviceSafetyLevel.CAUTION: ['modify', 'write', 'change', 'update'],
            DeviceSafetyLevel.QUARANTINED: ['wipe', 'erase', 'destroy', 'kill', 'blackout', 'hidden'],
        }

    def scan_smbios_tokens(self) -> List[Tuple[int, str]]:
        """Scan SMBIOS tokens for DSMIL devices"""
        logger.info("Scanning SMBIOS tokens...")
        discovered = []

        try:
            # Run smbios-token-ctl to get all tokens
            result = subprocess.run(
                ['sudo', 'smbios-token-ctl', '--dump-tokens'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse output for token information
                lines = result.stdout.split('\n')
                for line in lines:
                    # Look for token lines: "Token: 0x8000 (32768)"
                    match = re.search(r'Token:\s+(0x[0-9a-fA-F]+)\s+\((\d+)\)', line)
                    if match:
                        token_hex = match.group(1)
                        token_dec = int(token_hex, 16)

                        # Check if in DSMIL ranges
                        for start, end, range_name in self.DSMIL_RANGES:
                            if start <= token_dec <= end:
                                # Get token value
                                value_result = subprocess.run(
                                    ['sudo', 'smbios-token-ctl', f'--get-token={token_hex}'],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                value = value_result.stdout.strip() if value_result.returncode == 0 else "unknown"
                                discovered.append((token_dec, value))
                                logger.info(f"Found SMBIOS token {token_hex} = {value}")
                                break

            logger.info(f"SMBIOS scan complete: {len(discovered)} devices found")

        except subprocess.TimeoutExpired:
            logger.error("SMBIOS scan timeout")
        except FileNotFoundError:
            logger.warning("smbios-token-ctl not found - install libsmbios-bin")
        except Exception as e:
            logger.error(f"SMBIOS scan error: {e}")

        return discovered

    def scan_acpi_devices(self) -> List[Tuple[int, str]]:
        """Scan ACPI namespace for DSMIL devices"""
        logger.info("Scanning ACPI devices...")
        discovered = []

        try:
            acpi_paths = [
                '/sys/firmware/acpi/tables/DSDT',
                '/sys/firmware/acpi/tables/SSDT*',
            ]

            for path_pattern in acpi_paths:
                import glob
                for path in glob.glob(path_pattern):
                    try:
                        # Read ACPI table (requires root)
                        with open(path, 'rb') as f:
                            data = f.read(1024)  # Read first 1KB

                            # Look for DSMIL signatures
                            if b'DSMIL' in data or b'DELL' in data:
                                # Extract potential device IDs
                                for i in range(0, len(data) - 4, 4):
                                    value = struct.unpack('<I', data[i:i+4])[0]
                                    # Check if value looks like a DSMIL token
                                    for start, end, _ in self.DSMIL_RANGES:
                                        if start <= value <= end:
                                            discovered.append((value, 'acpi'))
                                            logger.info(f"Found ACPI device 0x{value:04X}")
                                            break
                    except PermissionError:
                        logger.warning(f"No permission to read {path}")
                    except Exception as e:
                        logger.debug(f"Error reading {path}: {e}")

            logger.info(f"ACPI scan complete: {len(discovered)} devices found")

        except Exception as e:
            logger.error(f"ACPI scan error: {e}")

        return discovered

    def scan_sysfs_interfaces(self) -> List[Tuple[int, str]]:
        """Scan sysfs for DSMIL device interfaces"""
        logger.info("Scanning sysfs interfaces...")
        discovered = []

        sysfs_paths = [
            '/sys/devices/platform/dell-milspec',
            '/sys/class/dmi',
            '/sys/firmware/dmi',
        ]

        for base_path in sysfs_paths:
            if Path(base_path).exists():
                try:
                    # Recursively search for device entries
                    for entry in Path(base_path).rglob('*'):
                        if entry.is_file():
                            # Look for device ID patterns in filenames
                            match = re.search(r'(0x[0-9a-fA-F]{4}|device_[0-9a-fA-F]{4})', entry.name)
                            if match:
                                device_id_str = match.group(1).replace('device_', '0x')
                                device_id = int(device_id_str, 16)
                                discovered.append((device_id, 'sysfs'))
                                logger.info(f"Found sysfs device 0x{device_id:04X}")
                except Exception as e:
                    logger.debug(f"Error scanning {base_path}: {e}")

        logger.info(f"sysfs scan complete: {len(discovered)} devices found")
        return discovered

    def predict_device_capabilities(self, device_id: int, signature: str) -> Tuple[List[str], DeviceSafetyLevel, float]:
        """Use ML to predict device capabilities and safety level"""

        # Simple pattern matching classifier (in production this would be a trained ML model)
        capabilities = []
        safety_scores = {level: 0.0 for level in DeviceSafetyLevel}

        signature_lower = signature.lower()

        # Classify by function
        for category, keywords in self.device_patterns.items():
            if any(kw in signature_lower for kw in keywords):
                capabilities.append(category)

        # Determine safety level
        for level, keywords in self.safety_patterns.items():
            score = sum(1 for kw in keywords if kw in signature_lower)
            safety_scores[level] = score

        # Get highest scoring safety level
        if safety_scores[DeviceSafetyLevel.QUARANTINED] > 0:
            safety_level = DeviceSafetyLevel.QUARANTINED
            confidence = 0.95
        elif safety_scores[DeviceSafetyLevel.CAUTION] > 0:
            safety_level = DeviceSafetyLevel.CAUTION
            confidence = 0.70
        elif safety_scores[DeviceSafetyLevel.MONITORED] > 0:
            safety_level = DeviceSafetyLevel.MONITORED
            confidence = 0.80
        elif safety_scores[DeviceSafetyLevel.SAFE] > 0:
            safety_level = DeviceSafetyLevel.SAFE
            confidence = 0.90
        else:
            safety_level = DeviceSafetyLevel.UNKNOWN
            confidence = 0.50

        # Special rules for known address ranges
        if device_id in [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]:
            safety_level = DeviceSafetyLevel.QUARANTINED
            confidence = 1.0

        return capabilities, safety_level, confidence

    def estimate_thermal_impact(self, device_id: int, capabilities: List[str]) -> float:
        """Estimate thermal impact of device activation"""

        # Base thermal impact
        thermal = 0.5

        # Adjust based on capabilities
        thermal_multipliers = {
            'security': 1.2,
            'network': 1.5,
            'storage': 2.0,
            'thermal': 0.5,
            'power': 1.0,
            'emergency': 0.0,
        }

        for capability in capabilities:
            thermal *= thermal_multipliers.get(capability, 1.0)

        # Device ID based adjustment
        group = (device_id - 0x8000) // 16 if 0x8000 <= device_id <= 0x806B else 0
        thermal += group * 0.2

        return min(thermal, 10.0)  # Cap at 10°C

    def calculate_activation_priority(self, device_id: int, dependencies: List[int], safety_level: DeviceSafetyLevel) -> int:
        """Calculate activation priority (lower = activate first)"""

        # Base priority by safety level
        priority_base = {
            DeviceSafetyLevel.SAFE: 10,
            DeviceSafetyLevel.MONITORED: 50,
            DeviceSafetyLevel.CAUTION: 100,
            DeviceSafetyLevel.QUARANTINED: 9999,
            DeviceSafetyLevel.UNKNOWN: 500,
        }

        priority = priority_base[safety_level]

        # Adjust based on dependencies
        priority += len(dependencies) * 5

        # Master controller gets highest priority
        if device_id == 0x8000:
            priority = 1

        return priority

    def discover_all_devices(self) -> Dict[int, HardwareDevice]:
        """Comprehensive hardware discovery using all methods"""
        logger.info("=" * 70)
        logger.info("Starting ML-Enhanced Hardware Discovery")
        logger.info("=" * 70)

        all_devices: Dict[int, Tuple[HardwareInterface, str]] = {}

        # Scan via multiple interfaces
        smbios_devices = self.scan_smbios_tokens()
        acpi_devices = self.scan_acpi_devices()
        sysfs_devices = self.scan_sysfs_interfaces()

        # Combine results
        for device_id, value in smbios_devices:
            all_devices[device_id] = (HardwareInterface.SMBIOS, value)

        for device_id, source in acpi_devices:
            if device_id not in all_devices:
                all_devices[device_id] = (HardwareInterface.ACPI, source)

        for device_id, source in sysfs_devices:
            if device_id not in all_devices:
                all_devices[device_id] = (HardwareInterface.SYSFS, source)

        logger.info(f"Total unique devices found: {len(all_devices)}")

        # ML classification and analysis
        logger.info("Performing ML classification...")

        for device_id, (interface, signature) in all_devices.items():
            # Generate device name
            if 0x8000 <= device_id <= 0x806B:
                group = (device_id - 0x8000) // 16
                dev_in_group = (device_id - 0x8000) % 16
                name = f"DSMIL{group}D{dev_in_group:X}"
            else:
                name = f"DEVICE_0x{device_id:04X}"

            # ML prediction
            capabilities, safety_level, confidence = self.predict_device_capabilities(device_id, signature)

            # Estimate thermal impact
            thermal_estimate = self.estimate_thermal_impact(device_id, capabilities)

            # Determine dependencies (simplified)
            dependencies = []
            if device_id != 0x8000 and 0x8000 <= device_id <= 0x806B:
                dependencies.append(0x8000)  # Most devices depend on master controller

            # Calculate activation priority
            priority = self.calculate_activation_priority(device_id, dependencies, safety_level)

            # Create hardware device record
            hw_device = HardwareDevice(
                address=device_id,
                name=name,
                interface=interface,
                signature=signature,
                capabilities=capabilities,
                safety_level=safety_level,
                ml_confidence=confidence,
                dependencies=dependencies,
                activation_priority=priority,
                thermal_estimate=thermal_estimate
            )

            self.discovered_devices[device_id] = hw_device

            logger.info(f"Device 0x{device_id:04X}: {name} | Safety: {safety_level.value} | "
                       f"Confidence: {confidence:.2f} | Priority: {priority}")

        logger.info("=" * 70)
        logger.info(f"Discovery complete: {len(self.discovered_devices)} devices classified")
        logger.info("=" * 70)

        return self.discovered_devices

    def get_activation_sequence(self) -> List[int]:
        """Get optimal activation sequence based on ML analysis"""

        # Sort devices by activation priority
        sorted_devices = sorted(
            self.discovered_devices.values(),
            key=lambda d: (d.activation_priority, d.address)
        )

        # Filter out quarantined devices
        safe_sequence = [
            d.address for d in sorted_devices
            if d.safety_level != DeviceSafetyLevel.QUARANTINED
        ]

        logger.info(f"Generated activation sequence: {len(safe_sequence)} devices")
        return safe_sequence

    def export_discovery_report(self, output_path: Optional[Path] = None) -> Dict:
        """Export comprehensive discovery report"""

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_devices': len(self.discovered_devices),
            'safety_breakdown': {},
            'interface_breakdown': {},
            'capability_breakdown': {},
            'activation_sequence': self.get_activation_sequence(),
            'devices': []
        }

        # Count by safety level
        for level in DeviceSafetyLevel:
            count = sum(1 for d in self.discovered_devices.values() if d.safety_level == level)
            report['safety_breakdown'][level.value] = count

        # Count by interface
        for interface in HardwareInterface:
            count = sum(1 for d in self.discovered_devices.values() if d.interface == interface)
            report['interface_breakdown'][interface.value] = count

        # Count capabilities
        capability_counts = {}
        for device in self.discovered_devices.values():
            for cap in device.capabilities:
                capability_counts[cap] = capability_counts.get(cap, 0) + 1
        report['capability_breakdown'] = capability_counts

        # Device details
        for device in sorted(self.discovered_devices.values(), key=lambda d: d.address):
            report['devices'].append({
                'address': f"0x{device.address:04X}",
                'name': device.name,
                'interface': device.interface.value,
                'capabilities': device.capabilities,
                'safety_level': device.safety_level.value,
                'ml_confidence': device.ml_confidence,
                'dependencies': [f"0x{d:04X}" for d in device.dependencies],
                'activation_priority': device.activation_priority,
                'thermal_estimate': device.thermal_estimate,
            })

        # Save report
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Discovery report saved to {output_path}")

        return report


def main():
    """Main entry point"""
    print("=" * 70)
    print("DSMIL ML-Enhanced Hardware Discovery")
    print("=" * 70)
    print()

    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root. Some scans may be limited.")
        print("   Run with: sudo python3 dsmil_ml_discovery.py")
        print()

    # Initialize discovery system
    discovery = DSMILMLDiscovery()

    # Run comprehensive discovery
    print("Starting comprehensive hardware scan...")
    print("This will scan SMBIOS, ACPI, and sysfs interfaces...")
    print()

    devices = discovery.discover_all_devices()

    # Generate activation sequence
    sequence = discovery.get_activation_sequence()

    # Export report
    report_path = Path('/tmp/dsmil_ml_discovery_report.json')
    report = discovery.export_discovery_report(report_path)

    # Print summary
    print()
    print("=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Total Devices: {report['total_devices']}")
    print()
    print("Safety Breakdown:")
    for level, count in report['safety_breakdown'].items():
        print(f"  {level.upper():<15}: {count:>3} devices")
    print()
    print("Interface Breakdown:")
    for interface, count in report['interface_breakdown'].items():
        print(f"  {interface.upper():<15}: {count:>3} devices")
    print()
    print(f"Activation Sequence: {len(sequence)} devices ready")
    print(f"Report saved to: {report_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

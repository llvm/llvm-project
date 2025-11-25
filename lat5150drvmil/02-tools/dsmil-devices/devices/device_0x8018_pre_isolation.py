#!/usr/bin/env python3
"""
Device 0x8018: Pre-Isolation

Network pre-isolation and threat containment system for compromised systems.
Provides graduated isolation levels before full quarantine or system lockdown.

Device ID: 0x8018
Group: 1 (Extended Security)
Risk Level: MONITORED (Security-critical isolation control)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class IsolationLevel:
    """System isolation levels"""
    NONE = 0
    MONITORING = 1
    RESTRICTED = 2
    QUARANTINE = 3
    FULL_ISOLATION = 4


class ThreatLevel:
    """Threat assessment levels"""
    CLEAN = 0
    SUSPICIOUS = 1
    PROBABLE = 2
    CONFIRMED = 3
    CRITICAL = 4


class IsolationReason:
    """Reasons for isolation"""
    MANUAL = 0
    MALWARE_DETECTED = 1
    ANOMALOUS_BEHAVIOR = 2
    POLICY_VIOLATION = 3
    FAILED_VALIDATION = 4
    EXTERNAL_THREAT = 5
    COMPROMISE_SUSPECTED = 6


class NetworkZone:
    """Network isolation zones"""
    OPERATIONAL = 0
    RESTRICTED = 1
    QUARANTINE = 2
    ISOLATION = 3
    MANAGEMENT_ONLY = 4


class PreIsolationDevice(DSMILDeviceBase):
    """Pre-Isolation (0x8018)"""

    # Register map
    REG_ISOLATION_STATUS = 0x00
    REG_ISOLATION_LEVEL = 0x04
    REG_THREAT_LEVEL = 0x08
    REG_NETWORK_ZONE = 0x0C
    REG_ISOLATED_COUNT = 0x10
    REG_QUARANTINE_COUNT = 0x14
    REG_LAST_ISOLATION = 0x18
    REG_AUTO_ISOLATION = 0x1C

    # Status bits
    STATUS_ISOLATION_ACTIVE = 0x01
    STATUS_MONITORING_ENABLED = 0x02
    STATUS_AUTO_ISOLATION = 0x04
    STATUS_THREAT_DETECTED = 0x08
    STATUS_QUARANTINE_ACTIVE = 0x10
    STATUS_ROLLBACK_AVAILABLE = 0x20

    def __init__(self, device_id: int = 0x8018,
                 name: str = "Pre-Isolation",
                 description: str = "Network Pre-Isolation and Threat Containment"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.isolation_level = IsolationLevel.NONE
        self.threat_level = ThreatLevel.CLEAN
        self.network_zone = NetworkZone.OPERATIONAL

        self.isolation_active = False
        self.monitoring_enabled = True
        self.auto_isolation_enabled = True

        self.isolated_systems = {}
        self.quarantine_history = []
        self.max_history = 500

        self.isolation_count = 0
        self.quarantine_count = 0
        self.last_isolation_time = 0

        # Initialize with sample data
        self._initialize_sample_data()

        # Register map
        self.register_map = {
            "ISOLATION_STATUS": {
                "offset": self.REG_ISOLATION_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Isolation system status"
            },
            "ISOLATION_LEVEL": {
                "offset": self.REG_ISOLATION_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Current isolation level"
            },
            "THREAT_LEVEL": {
                "offset": self.REG_THREAT_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Current threat assessment level"
            },
            "NETWORK_ZONE": {
                "offset": self.REG_NETWORK_ZONE,
                "size": 4,
                "access": "RO",
                "description": "Current network zone"
            },
            "ISOLATED_COUNT": {
                "offset": self.REG_ISOLATED_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of isolated systems"
            },
            "QUARANTINE_COUNT": {
                "offset": self.REG_QUARANTINE_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of quarantined systems"
            },
            "LAST_ISOLATION": {
                "offset": self.REG_LAST_ISOLATION,
                "size": 4,
                "access": "RO",
                "description": "Last isolation timestamp"
            },
            "AUTO_ISOLATION": {
                "offset": self.REG_AUTO_ISOLATION,
                "size": 4,
                "access": "RO",
                "description": "Auto-isolation enabled"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Pre-Isolation device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize isolation system
            self._initialize_sample_data()

            self.isolation_level = IsolationLevel.NONE
            self.threat_level = ThreatLevel.CLEAN
            self.network_zone = NetworkZone.OPERATIONAL

            self.isolation_active = False
            self.monitoring_enabled = True
            self.auto_isolation_enabled = True

            self.isolation_count = 0
            self.quarantine_count = 0
            self.last_isolation_time = 0
            self.quarantine_history = []

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "isolation_level": self._get_isolation_level_name(self.isolation_level),
                "threat_level": self._get_threat_level_name(self.threat_level),
                "monitoring_enabled": self.monitoring_enabled,
                "auto_isolation": self.auto_isolation_enabled,
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.EVENT_MONITORING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "isolation_active": bool(status_reg & self.STATUS_ISOLATION_ACTIVE),
            "monitoring_enabled": bool(status_reg & self.STATUS_MONITORING_ENABLED),
            "auto_isolation": bool(status_reg & self.STATUS_AUTO_ISOLATION),
            "threat_detected": bool(status_reg & self.STATUS_THREAT_DETECTED),
            "quarantine_active": bool(status_reg & self.STATUS_QUARANTINE_ACTIVE),
            "rollback_available": bool(status_reg & self.STATUS_ROLLBACK_AVAILABLE),
            "isolation_level": self._get_isolation_level_name(self.isolation_level),
            "threat_level": self._get_threat_level_name(self.threat_level),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "ISOLATION_STATUS":
                value = self._read_status_register()
            elif register == "ISOLATION_LEVEL":
                value = self.isolation_level
            elif register == "THREAT_LEVEL":
                value = self.threat_level
            elif register == "NETWORK_ZONE":
                value = self.network_zone
            elif register == "ISOLATED_COUNT":
                value = len(self.isolated_systems)
            elif register == "QUARANTINE_COUNT":
                value = self.quarantine_count
            elif register == "LAST_ISOLATION":
                value = self.last_isolation_time
            elif register == "AUTO_ISOLATION":
                value = 1 if self.auto_isolation_enabled else 0
            else:
                value = 0

            self._record_operation(True)
            return OperationResult(True, data={
                "register": register,
                "value": value,
                "hex": f"0x{value:08X}",
            })

        except Exception as e:
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    # Pre-Isolation specific operations

    def get_isolation_status(self) -> OperationResult:
        """Get comprehensive isolation status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        status = {
            "isolation_level": self._get_isolation_level_name(self.isolation_level),
            "threat_level": self._get_threat_level_name(self.threat_level),
            "network_zone": self._get_network_zone_name(self.network_zone),
            "isolation_active": self.isolation_active,
            "monitoring_enabled": self.monitoring_enabled,
            "auto_isolation": self.auto_isolation_enabled,
            "isolated_systems": len(self.isolated_systems),
            "quarantine_count": self.quarantine_count,
        }

        self._record_operation(True)
        return OperationResult(True, data=status)

    def list_isolated_systems(self) -> OperationResult:
        """List all isolated systems"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        systems_list = []
        for system_id, system_info in self.isolated_systems.items():
            systems_list.append({
                "id": system_id,
                "isolation_level": self._get_isolation_level_name(system_info['isolation_level']),
                "reason": self._get_isolation_reason_name(system_info['reason']),
                "threat_level": self._get_threat_level_name(system_info['threat_level']),
                "isolated_at": system_info['isolated_at'],
                "network_zone": self._get_network_zone_name(system_info['zone']),
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "systems": systems_list,
            "total": len(systems_list),
        })

    def get_quarantine_history(self, limit: int = 50) -> OperationResult:
        """Get quarantine action history"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        history = self.quarantine_history[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "history": history,
            "total": len(history),
            "showing": f"Last {min(limit, len(self.quarantine_history))} of {len(self.quarantine_history)}",
        })

    def get_threat_assessment(self) -> OperationResult:
        """Get current threat assessment"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        assessment = {
            "overall_threat_level": self._get_threat_level_name(self.threat_level),
            "isolated_systems": len(self.isolated_systems),
            "quarantined_systems": self.quarantine_count,
            "monitoring_active": self.monitoring_enabled,
            "auto_isolation_enabled": self.auto_isolation_enabled,
            "recommended_action": self._get_recommended_action(),
        }

        self._record_operation(True)
        return OperationResult(True, data=assessment)

    def get_network_zones(self) -> OperationResult:
        """Get network zone configuration"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Count systems per zone
        zone_counts = {}
        for system_info in self.isolated_systems.values():
            zone = self._get_network_zone_name(system_info['zone'])
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        zones = {
            "current_zone": self._get_network_zone_name(self.network_zone),
            "systems_by_zone": zone_counts,
            "available_zones": [
                self._get_network_zone_name(NetworkZone.OPERATIONAL),
                self._get_network_zone_name(NetworkZone.RESTRICTED),
                self._get_network_zone_name(NetworkZone.QUARANTINE),
                self._get_network_zone_name(NetworkZone.ISOLATION),
            ],
        }

        self._record_operation(True)
        return OperationResult(True, data=zones)

    def get_isolation_summary(self) -> OperationResult:
        """Get isolation system summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Count by isolation level
        by_level = {}
        for system_info in self.isolated_systems.values():
            level = self._get_isolation_level_name(system_info['isolation_level'])
            by_level[level] = by_level.get(level, 0) + 1

        # Count by reason
        by_reason = {}
        for system_info in self.isolated_systems.values():
            reason = self._get_isolation_reason_name(system_info['reason'])
            by_reason[reason] = by_reason.get(reason, 0) + 1

        summary = {
            "total_isolated": len(self.isolated_systems),
            "total_quarantined": self.quarantine_count,
            "by_isolation_level": by_level,
            "by_reason": by_reason,
            "auto_isolation_enabled": self.auto_isolation_enabled,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pre-isolation statistics"""
        stats = super().get_statistics()

        stats.update({
            "isolation_level": self._get_isolation_level_name(self.isolation_level),
            "threat_level": self._get_threat_level_name(self.threat_level),
            "isolated_systems": len(self.isolated_systems),
            "quarantine_count": self.quarantine_count,
            "auto_isolation_enabled": self.auto_isolation_enabled,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read isolation status register (simulated)"""
        status = 0

        if self.isolation_active:
            status |= self.STATUS_ISOLATION_ACTIVE

        if self.monitoring_enabled:
            status |= self.STATUS_MONITORING_ENABLED

        if self.auto_isolation_enabled:
            status |= self.STATUS_AUTO_ISOLATION

        if self.threat_level >= ThreatLevel.SUSPICIOUS:
            status |= self.STATUS_THREAT_DETECTED

        if self.quarantine_count > 0:
            status |= self.STATUS_QUARANTINE_ACTIVE

        # Rollback available if isolation is active
        if self.isolation_active:
            status |= self.STATUS_ROLLBACK_AVAILABLE

        return status

    def _initialize_sample_data(self):
        """Initialize with sample isolated systems"""
        current_time = int(time.time())

        self.isolated_systems = {
            "sys_001": {
                "isolation_level": IsolationLevel.MONITORING,
                "threat_level": ThreatLevel.SUSPICIOUS,
                "reason": IsolationReason.ANOMALOUS_BEHAVIOR,
                "zone": NetworkZone.RESTRICTED,
                "isolated_at": current_time - 3600,
            },
            "sys_002": {
                "isolation_level": IsolationLevel.QUARANTINE,
                "threat_level": ThreatLevel.PROBABLE,
                "reason": IsolationReason.MALWARE_DETECTED,
                "zone": NetworkZone.QUARANTINE,
                "isolated_at": current_time - 7200,
            },
        }

    def _get_recommended_action(self) -> str:
        """Get recommended action based on threat level"""
        if self.threat_level >= ThreatLevel.CRITICAL:
            return "Immediate full isolation required"
        elif self.threat_level >= ThreatLevel.CONFIRMED:
            return "Quarantine recommended"
        elif self.threat_level >= ThreatLevel.PROBABLE:
            return "Enhanced monitoring recommended"
        elif self.threat_level >= ThreatLevel.SUSPICIOUS:
            return "Restrict network access"
        else:
            return "No action required"

    def _get_isolation_level_name(self, level: int) -> str:
        """Get isolation level name"""
        names = {
            IsolationLevel.NONE: "None",
            IsolationLevel.MONITORING: "Monitoring",
            IsolationLevel.RESTRICTED: "Restricted",
            IsolationLevel.QUARANTINE: "Quarantine",
            IsolationLevel.FULL_ISOLATION: "Full Isolation",
        }
        return names.get(level, "Unknown")

    def _get_threat_level_name(self, level: int) -> str:
        """Get threat level name"""
        names = {
            ThreatLevel.CLEAN: "Clean",
            ThreatLevel.SUSPICIOUS: "Suspicious",
            ThreatLevel.PROBABLE: "Probable",
            ThreatLevel.CONFIRMED: "Confirmed",
            ThreatLevel.CRITICAL: "Critical",
        }
        return names.get(level, "Unknown")

    def _get_isolation_reason_name(self, reason: int) -> str:
        """Get isolation reason name"""
        names = {
            IsolationReason.MANUAL: "Manual",
            IsolationReason.MALWARE_DETECTED: "Malware Detected",
            IsolationReason.ANOMALOUS_BEHAVIOR: "Anomalous Behavior",
            IsolationReason.POLICY_VIOLATION: "Policy Violation",
            IsolationReason.FAILED_VALIDATION: "Failed Validation",
            IsolationReason.EXTERNAL_THREAT: "External Threat",
            IsolationReason.COMPROMISE_SUSPECTED: "Compromise Suspected",
        }
        return names.get(reason, "Unknown")

    def _get_network_zone_name(self, zone: int) -> str:
        """Get network zone name"""
        names = {
            NetworkZone.OPERATIONAL: "Operational",
            NetworkZone.RESTRICTED: "Restricted",
            NetworkZone.QUARANTINE: "Quarantine",
            NetworkZone.ISOLATION: "Isolation",
            NetworkZone.MANAGEMENT_ONLY: "Management Only",
        }
        return names.get(zone, "Unknown")


def main():
    """Test Pre-Isolation device"""
    print("=" * 80)
    print("Device 0x8018: Pre-Isolation - Test")
    print("=" * 80)

    device = PreIsolationDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Isolation level: {result.data['isolation_level']}")
        print(f"   Threat level: {result.data['threat_level']}")
        print(f"   Auto-isolation: {result.data['auto_isolation']}")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   Monitoring enabled: {status['monitoring_enabled']}")
    print(f"   Isolation level: {status['isolation_level']}")
    print(f"   Threat level: {status['threat_level']}")

    # Get isolation status
    print("\n3. Getting isolation status...")
    result = device.get_isolation_status()
    if result.success:
        print(f"   Isolation level: {result.data['isolation_level']}")
        print(f"   Network zone: {result.data['network_zone']}")
        print(f"   Isolated systems: {result.data['isolated_systems']}")

    # List isolated systems
    print("\n4. Listing isolated systems...")
    result = device.list_isolated_systems()
    if result.success:
        print(f"   Total isolated: {result.data['total']}")
        for system in result.data['systems']:
            print(f"   - {system['id']}: {system['isolation_level']} ({system['reason']})")

    # Get threat assessment
    print("\n5. Getting threat assessment...")
    result = device.get_threat_assessment()
    if result.success:
        print(f"   Threat level: {result.data['overall_threat_level']}")
        print(f"   Recommended action: {result.data['recommended_action']}")

    # Get isolation summary
    print("\n6. Getting isolation summary...")
    result = device.get_isolation_summary()
    if result.success:
        print(f"   Total isolated: {result.data['total_isolated']}")
        print(f"   By level: {result.data['by_isolation_level']}")
        print(f"   By reason: {result.data['by_reason']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

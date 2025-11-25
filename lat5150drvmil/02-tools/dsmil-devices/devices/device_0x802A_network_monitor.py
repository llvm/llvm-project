#!/usr/bin/env python3
"""
Device 0x802A: Network Monitor

Network traffic monitoring and analysis for intrusion detection and performance.
Provides real-time network monitoring, protocol analysis, and anomaly detection.

Device ID: 0x802A
Group: 2 (Network/Communications)
Risk Level: SAFE (Read-only network monitoring)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time
import random

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class ProtocolType:
    """Network protocol types"""
    TCP = 0
    UDP = 1
    ICMP = 2
    HTTP = 3
    HTTPS = 4
    SSH = 5
    DNS = 6
    FTP = 7


class TrafficDirection:
    """Traffic direction"""
    INBOUND = 0
    OUTBOUND = 1
    INTERNAL = 2


class AlertSeverity:
    """Network alert severity"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MonitoringMode:
    """Network monitoring modes"""
    PASSIVE = 0
    ACTIVE = 1
    DEEP_INSPECTION = 2
    IDS_IPS = 3


class NetworkMonitorDevice(DSMILDeviceBase):
    """Network Monitor (0x802A)"""

    # Register map
    REG_MONITOR_STATUS = 0x00
    REG_PACKET_COUNT = 0x04
    REG_BYTE_COUNT = 0x08
    REG_BANDWIDTH_USAGE = 0x0C
    REG_ALERT_COUNT = 0x10
    REG_ANOMALY_COUNT = 0x14
    REG_MONITORING_MODE = 0x18
    REG_INTERFACE_COUNT = 0x1C

    # Status bits
    STATUS_MONITORING_ACTIVE = 0x01
    STATUS_IDS_ENABLED = 0x02
    STATUS_ANOMALY_DETECTION = 0x04
    STATUS_ALERTS_ACTIVE = 0x08
    STATUS_DEEP_INSPECTION = 0x10
    STATUS_PCAP_ENABLED = 0x20

    def __init__(self, device_id: int = 0x802A,
                 name: str = "Network Monitor",
                 description: str = "Network Traffic Monitoring and Analysis"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.interfaces = {}
        self.traffic_stats = {}
        self.alerts = []
        self.max_alerts = 1000

        self.monitoring_active = True
        self.monitoring_mode = MonitoringMode.PASSIVE
        self.ids_enabled = True
        self.anomaly_detection_enabled = True
        self.deep_inspection_enabled = False
        self.pcap_enabled = False

        self.total_packets = 0
        self.total_bytes = 0
        self.alert_count = 0
        self.anomaly_count = 0

        # Initialize interfaces and stats
        self._initialize_interfaces()
        self._update_traffic_stats()

        # Register map
        self.register_map = {
            "MONITOR_STATUS": {
                "offset": self.REG_MONITOR_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Network monitor status"
            },
            "PACKET_COUNT": {
                "offset": self.REG_PACKET_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total packet count"
            },
            "BYTE_COUNT": {
                "offset": self.REG_BYTE_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total byte count"
            },
            "BANDWIDTH_USAGE": {
                "offset": self.REG_BANDWIDTH_USAGE,
                "size": 4,
                "access": "RO",
                "description": "Current bandwidth usage (Mbps)"
            },
            "ALERT_COUNT": {
                "offset": self.REG_ALERT_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total alert count"
            },
            "ANOMALY_COUNT": {
                "offset": self.REG_ANOMALY_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Anomaly detection count"
            },
            "MONITORING_MODE": {
                "offset": self.REG_MONITORING_MODE,
                "size": 4,
                "access": "RO",
                "description": "Current monitoring mode"
            },
            "INTERFACE_COUNT": {
                "offset": self.REG_INTERFACE_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of monitored interfaces"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Network Monitor device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize monitoring system
            self._initialize_interfaces()

            self.monitoring_active = True
            self.monitoring_mode = MonitoringMode.PASSIVE
            self.ids_enabled = True
            self.anomaly_detection_enabled = True
            self.deep_inspection_enabled = False
            self.pcap_enabled = False

            self.total_packets = 0
            self.total_bytes = 0
            self.alert_count = 0
            self.anomaly_count = 0
            self.alerts = []

            # Update initial stats
            self._update_traffic_stats()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "monitoring_active": self.monitoring_active,
                "interfaces": len(self.interfaces),
                "monitoring_mode": self._get_monitoring_mode_name(self.monitoring_mode),
                "ids_enabled": self.ids_enabled,
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
            "monitoring_active": bool(status_reg & self.STATUS_MONITORING_ACTIVE),
            "ids_enabled": bool(status_reg & self.STATUS_IDS_ENABLED),
            "anomaly_detection": bool(status_reg & self.STATUS_ANOMALY_DETECTION),
            "alerts_active": bool(status_reg & self.STATUS_ALERTS_ACTIVE),
            "deep_inspection": bool(status_reg & self.STATUS_DEEP_INSPECTION),
            "pcap_enabled": bool(status_reg & self.STATUS_PCAP_ENABLED),
            "total_packets": self.total_packets,
            "alert_count": self.alert_count,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "MONITOR_STATUS":
                value = self._read_status_register()
            elif register == "PACKET_COUNT":
                value = self.total_packets
            elif register == "BYTE_COUNT":
                value = self.total_bytes
            elif register == "BANDWIDTH_USAGE":
                value = self._calculate_bandwidth()
            elif register == "ALERT_COUNT":
                value = self.alert_count
            elif register == "ANOMALY_COUNT":
                value = self.anomaly_count
            elif register == "MONITORING_MODE":
                value = self.monitoring_mode
            elif register == "INTERFACE_COUNT":
                value = len(self.interfaces)
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

    # Network Monitor specific operations

    def list_interfaces(self) -> OperationResult:
        """List all monitored network interfaces"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        interfaces_list = []
        for iface_id, iface_info in self.interfaces.items():
            interfaces_list.append({
                "name": iface_info['name'],
                "status": iface_info['status'],
                "speed_mbps": iface_info['speed'],
                "packets_rx": iface_info['packets_rx'],
                "packets_tx": iface_info['packets_tx'],
                "bytes_rx": iface_info['bytes_rx'],
                "bytes_tx": iface_info['bytes_tx'],
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "interfaces": interfaces_list,
            "total": len(interfaces_list),
        })

    def get_traffic_stats(self) -> OperationResult:
        """Get overall traffic statistics"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_traffic_stats()

        stats = {
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "total_bytes_gb": round(self.total_bytes / (1024**3), 2),
            "bandwidth_mbps": self._calculate_bandwidth(),
            "by_protocol": self.traffic_stats.get('by_protocol', {}),
            "by_direction": self.traffic_stats.get('by_direction', {}),
        }

        self._record_operation(True)
        return OperationResult(True, data=stats)

    def get_protocol_breakdown(self) -> OperationResult:
        """Get traffic breakdown by protocol"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_traffic_stats()

        breakdown = self.traffic_stats.get('by_protocol', {})

        self._record_operation(True)
        return OperationResult(True, data={
            "protocols": breakdown,
            "total_packets": sum(breakdown.values()),
        })

    def get_alerts(self, severity_filter: Optional[int] = None, limit: int = 50) -> OperationResult:
        """Get network alerts"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        alerts_list = self.alerts
        if severity_filter is not None:
            alerts_list = [a for a in alerts_list if a['severity'] == severity_filter]

        alerts_list = alerts_list[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "alerts": alerts_list,
            "total": len(alerts_list),
            "showing": f"Last {min(limit, len(alerts_list))} of {len(self.alerts)}",
        })

    def get_bandwidth_usage(self) -> OperationResult:
        """Get current bandwidth usage"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        bandwidth = self._calculate_bandwidth()

        # Per-interface bandwidth
        by_interface = {}
        for iface_id, iface_info in self.interfaces.items():
            by_interface[iface_info['name']] = {
                "current_mbps": round(random.uniform(10, 100), 2),
                "capacity_mbps": iface_info['speed'],
                "utilization": round(random.uniform(10, 80), 1),
            }

        self._record_operation(True)
        return OperationResult(True, data={
            "total_mbps": bandwidth,
            "by_interface": by_interface,
        })

    def get_anomalies(self, limit: int = 50) -> OperationResult:
        """Get detected network anomalies"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Sample anomalies
        anomalies = [
            {
                "timestamp": time.time() - 3600,
                "type": "Port Scan",
                "source": "10.1.100.50",
                "severity": "Medium",
                "description": "Sequential port scanning detected",
            },
            {
                "timestamp": time.time() - 1800,
                "type": "Unusual Traffic Volume",
                "source": "10.1.50.25",
                "severity": "Low",
                "description": "Traffic volume 200% above baseline",
            },
        ]

        anomalies = anomalies[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "anomalies": anomalies,
            "total": len(anomalies),
            "detection_enabled": self.anomaly_detection_enabled,
        })

    def get_monitoring_config(self) -> OperationResult:
        """Get network monitoring configuration"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        config = {
            "monitoring_mode": self._get_monitoring_mode_name(self.monitoring_mode),
            "ids_enabled": self.ids_enabled,
            "anomaly_detection": self.anomaly_detection_enabled,
            "deep_inspection": self.deep_inspection_enabled,
            "pcap_enabled": self.pcap_enabled,
            "interfaces_monitored": len(self.interfaces),
        }

        self._record_operation(True)
        return OperationResult(True, data=config)

    def get_summary(self) -> OperationResult:
        """Get comprehensive network monitoring summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_traffic_stats()

        summary = {
            "monitoring_active": self.monitoring_active,
            "interfaces": len(self.interfaces),
            "total_packets": self.total_packets,
            "total_gb": round(self.total_bytes / (1024**3), 2),
            "bandwidth_mbps": self._calculate_bandwidth(),
            "alerts": self.alert_count,
            "anomalies": self.anomaly_count,
            "ids_enabled": self.ids_enabled,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_statistics(self) -> Dict[str, Any]:
        """Get network monitor statistics"""
        stats = super().get_statistics()

        self._update_traffic_stats()

        stats.update({
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "alert_count": self.alert_count,
            "anomaly_count": self.anomaly_count,
            "monitoring_active": self.monitoring_active,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read network monitor status register (simulated)"""
        status = 0

        if self.monitoring_active:
            status |= self.STATUS_MONITORING_ACTIVE

        if self.ids_enabled:
            status |= self.STATUS_IDS_ENABLED

        if self.anomaly_detection_enabled:
            status |= self.STATUS_ANOMALY_DETECTION

        if self.alert_count > 0:
            status |= self.STATUS_ALERTS_ACTIVE

        if self.deep_inspection_enabled:
            status |= self.STATUS_DEEP_INSPECTION

        if self.pcap_enabled:
            status |= self.STATUS_PCAP_ENABLED

        return status

    def _initialize_interfaces(self):
        """Initialize network interfaces"""
        self.interfaces = {
            "eth0": {
                "name": "eth0",
                "status": "up",
                "speed": 1000,  # Mbps
                "packets_rx": 0,
                "packets_tx": 0,
                "bytes_rx": 0,
                "bytes_tx": 0,
            },
            "eth1": {
                "name": "eth1",
                "status": "up",
                "speed": 1000,
                "packets_rx": 0,
                "packets_tx": 0,
                "bytes_rx": 0,
                "bytes_tx": 0,
            },
            "wlan0": {
                "name": "wlan0",
                "status": "down",
                "speed": 300,
                "packets_rx": 0,
                "packets_tx": 0,
                "bytes_rx": 0,
                "bytes_tx": 0,
            },
        }

    def _update_traffic_stats(self):
        """Update traffic statistics (simulated)"""
        # Simulate traffic
        self.total_packets = random.randint(100000, 500000)
        self.total_bytes = random.randint(1024**3, 10 * 1024**3)  # 1-10 GB

        # Update interface stats
        for iface_info in self.interfaces.values():
            if iface_info['status'] == 'up':
                iface_info['packets_rx'] = random.randint(10000, 100000)
                iface_info['packets_tx'] = random.randint(10000, 100000)
                iface_info['bytes_rx'] = random.randint(1024**2, 1024**3)
                iface_info['bytes_tx'] = random.randint(1024**2, 1024**3)

        # Protocol breakdown
        self.traffic_stats['by_protocol'] = {
            "TCP": random.randint(40000, 80000),
            "UDP": random.randint(10000, 30000),
            "HTTPS": random.randint(30000, 60000),
            "HTTP": random.randint(5000, 15000),
            "DNS": random.randint(5000, 10000),
            "SSH": random.randint(1000, 5000),
        }

        # Direction breakdown
        self.traffic_stats['by_direction'] = {
            "Inbound": random.randint(40000, 80000),
            "Outbound": random.randint(40000, 80000),
            "Internal": random.randint(10000, 30000),
        }

    def _calculate_bandwidth(self) -> int:
        """Calculate current bandwidth usage in Mbps"""
        return random.randint(50, 500)

    def _get_monitoring_mode_name(self, mode: int) -> str:
        """Get monitoring mode name"""
        names = {
            MonitoringMode.PASSIVE: "Passive",
            MonitoringMode.ACTIVE: "Active",
            MonitoringMode.DEEP_INSPECTION: "Deep Inspection",
            MonitoringMode.IDS_IPS: "IDS/IPS",
        }
        return names.get(mode, "Unknown")

    def _get_protocol_name(self, protocol: int) -> str:
        """Get protocol name"""
        names = {
            ProtocolType.TCP: "TCP",
            ProtocolType.UDP: "UDP",
            ProtocolType.ICMP: "ICMP",
            ProtocolType.HTTP: "HTTP",
            ProtocolType.HTTPS: "HTTPS",
            ProtocolType.SSH: "SSH",
            ProtocolType.DNS: "DNS",
            ProtocolType.FTP: "FTP",
        }
        return names.get(protocol, "Unknown")

    def _get_alert_severity_name(self, severity: int) -> str:
        """Get alert severity name"""
        names = {
            AlertSeverity.INFO: "Info",
            AlertSeverity.LOW: "Low",
            AlertSeverity.MEDIUM: "Medium",
            AlertSeverity.HIGH: "High",
            AlertSeverity.CRITICAL: "Critical",
        }
        return names.get(severity, "Unknown")


def main():
    """Test Network Monitor device"""
    print("=" * 80)
    print("Device 0x802A: Network Monitor - Test")
    print("=" * 80)

    device = NetworkMonitorDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Monitoring active: {result.data['monitoring_active']}")
        print(f"   Interfaces: {result.data['interfaces']}")
        print(f"   IDS enabled: {result.data['ids_enabled']}")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   Monitoring active: {status['monitoring_active']}")
    print(f"   IDS enabled: {status['ids_enabled']}")
    print(f"   Total packets: {status['total_packets']}")

    # List interfaces
    print("\n3. Listing network interfaces...")
    result = device.list_interfaces()
    if result.success:
        print(f"   Total interfaces: {result.data['total']}")
        for iface in result.data['interfaces']:
            print(f"   - {iface['name']}: {iface['status']} ({iface['speed_mbps']} Mbps)")

    # Get traffic stats
    print("\n4. Getting traffic statistics...")
    result = device.get_traffic_stats()
    if result.success:
        print(f"   Total packets: {result.data['total_packets']}")
        print(f"   Total data: {result.data['total_bytes_gb']} GB")
        print(f"   Bandwidth: {result.data['bandwidth_mbps']} Mbps")

    # Get protocol breakdown
    print("\n5. Getting protocol breakdown...")
    result = device.get_protocol_breakdown()
    if result.success:
        print(f"   Protocols detected:")
        for proto, count in list(result.data['protocols'].items())[:3]:
            print(f"      {proto}: {count} packets")

    # Get summary
    print("\n6. Getting monitoring summary...")
    result = device.get_summary()
    if result.success:
        print(f"   Interfaces: {result.data['interfaces']}")
        print(f"   Total packets: {result.data['total_packets']}")
        print(f"   Bandwidth: {result.data['bandwidth_mbps']} Mbps")
        print(f"   Alerts: {result.data['alerts']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

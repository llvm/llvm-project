#!/usr/bin/env python3
"""
Device 0x805A: Sensor Array Controller

Multi-sensor environmental and security monitoring system with fusion
capabilities for tactical situational awareness and threat detection.

Device ID: 0x805A
Group: 5 (Peripheral/Sensors)
Risk Level: SAFE (Read-only sensor monitoring)

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


class SensorType:
    """Sensor type identifiers"""
    TEMPERATURE = 0
    HUMIDITY = 1
    PRESSURE = 2
    MOTION = 3
    VIBRATION = 4
    LIGHT = 5
    ACOUSTIC = 6
    RADIATION = 7
    CHEMICAL = 8
    BIOLOGICAL = 9
    MAGNETIC = 10
    TAMPER = 11


class SensorStatus:
    """Sensor operational status"""
    OFFLINE = 0
    INITIALIZING = 1
    ONLINE = 2
    CALIBRATING = 3
    ERROR = 4
    DISABLED = 5


class AlertLevel:
    """Sensor alert levels"""
    NORMAL = 0
    INFO = 1
    WARNING = 2
    ALERT = 3
    CRITICAL = 4


class SensorArrayDevice(DSMILDeviceBase):
    """Sensor Array Controller (0x805A)"""

    # Register map
    REG_ARRAY_STATUS = 0x00
    REG_ACTIVE_SENSORS = 0x04
    REG_ALERT_STATUS = 0x08
    REG_TEMPERATURE = 0x0C
    REG_HUMIDITY = 0x10
    REG_PRESSURE = 0x14
    REG_MOTION_LEVEL = 0x18
    REG_LIGHT_LEVEL = 0x1C
    REG_RADIATION_LEVEL = 0x20
    REG_TAMPER_STATUS = 0x24

    # Status bits
    STATUS_ARRAY_ACTIVE = 0x01
    STATUS_CALIBRATED = 0x02
    STATUS_FUSION_ENABLED = 0x04
    STATUS_ALERT_ACTIVE = 0x08
    STATUS_TAMPER_DETECTED = 0x10
    STATUS_MOTION_DETECTED = 0x20
    STATUS_LOGGING_ENABLED = 0x40
    STATUS_ALL_SENSORS_OK = 0x80

    def __init__(self, device_id: int = 0x805A,
                 name: str = "Sensor Array Controller",
                 description: str = "Multi-Sensor Environmental and Security Monitoring"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.sensors = {}
        self.fusion_enabled = True
        self.alert_threshold = AlertLevel.WARNING
        self.sampling_rate = 1.0  # Hz
        self.sensor_history = []
        self.max_history = 500

        # Initialize sensors
        self._initialize_sensors()

        # Register map
        self.register_map = {
            "ARRAY_STATUS": {
                "offset": self.REG_ARRAY_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Sensor array status"
            },
            "ACTIVE_SENSORS": {
                "offset": self.REG_ACTIVE_SENSORS,
                "size": 4,
                "access": "RO",
                "description": "Number of active sensors"
            },
            "ALERT_STATUS": {
                "offset": self.REG_ALERT_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Combined alert status"
            },
            "TEMPERATURE": {
                "offset": self.REG_TEMPERATURE,
                "size": 4,
                "access": "RO",
                "description": "Temperature (°C x 100)"
            },
            "HUMIDITY": {
                "offset": self.REG_HUMIDITY,
                "size": 4,
                "access": "RO",
                "description": "Relative humidity (% x 100)"
            },
            "PRESSURE": {
                "offset": self.REG_PRESSURE,
                "size": 4,
                "access": "RO",
                "description": "Atmospheric pressure (hPa x 10)"
            },
            "MOTION_LEVEL": {
                "offset": self.REG_MOTION_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Motion detection level (0-255)"
            },
            "LIGHT_LEVEL": {
                "offset": self.REG_LIGHT_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Ambient light level (lux)"
            },
            "RADIATION_LEVEL": {
                "offset": self.REG_RADIATION_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Radiation level (μSv/h x 100)"
            },
            "TAMPER_STATUS": {
                "offset": self.REG_TAMPER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Tamper detection status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Sensor Array Controller"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize sensors
            self._initialize_sensors()
            self.fusion_enabled = True
            self.alert_threshold = AlertLevel.WARNING

            # Update all sensor readings
            self._update_all_sensors()

            self.state = DeviceState.READY
            self._record_operation(True)

            active_count = sum(1 for s in self.sensors.values()
                             if s["status"] == SensorStatus.ONLINE)

            return OperationResult(True, data={
                "total_sensors": len(self.sensors),
                "active_sensors": active_count,
                "fusion_enabled": self.fusion_enabled,
                "sampling_rate": self.sampling_rate,
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

        active_sensors = sum(1 for s in self.sensors.values()
                           if s["status"] == SensorStatus.ONLINE)

        return {
            "array_active": bool(status_reg & self.STATUS_ARRAY_ACTIVE),
            "calibrated": bool(status_reg & self.STATUS_CALIBRATED),
            "fusion_enabled": bool(status_reg & self.STATUS_FUSION_ENABLED),
            "alert_active": bool(status_reg & self.STATUS_ALERT_ACTIVE),
            "tamper_detected": bool(status_reg & self.STATUS_TAMPER_DETECTED),
            "motion_detected": bool(status_reg & self.STATUS_MOTION_DETECTED),
            "logging_enabled": bool(status_reg & self.STATUS_LOGGING_ENABLED),
            "all_sensors_ok": bool(status_reg & self.STATUS_ALL_SENSORS_OK),
            "active_sensors": active_sensors,
            "total_sensors": len(self.sensors),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            # Update sensors before reading
            self._update_all_sensors()

            if register == "ARRAY_STATUS":
                value = self._read_status_register()
            elif register == "ACTIVE_SENSORS":
                value = sum(1 for s in self.sensors.values()
                          if s["status"] == SensorStatus.ONLINE)
            elif register == "ALERT_STATUS":
                value = self._get_highest_alert_level()
            elif register == "TEMPERATURE":
                temp_sensor = self.sensors.get("temp0")
                value = int(temp_sensor["value"] * 100) if temp_sensor else 0
            elif register == "HUMIDITY":
                humid_sensor = self.sensors.get("humid0")
                value = int(humid_sensor["value"] * 100) if humid_sensor else 0
            elif register == "PRESSURE":
                press_sensor = self.sensors.get("press0")
                value = int(press_sensor["value"] * 10) if press_sensor else 0
            elif register == "MOTION_LEVEL":
                motion_sensor = self.sensors.get("motion0")
                value = int(motion_sensor["value"]) if motion_sensor else 0
            elif register == "LIGHT_LEVEL":
                light_sensor = self.sensors.get("light0")
                value = int(light_sensor["value"]) if light_sensor else 0
            elif register == "RADIATION_LEVEL":
                rad_sensor = self.sensors.get("rad0")
                value = int(rad_sensor["value"] * 100) if rad_sensor else 0
            elif register == "TAMPER_STATUS":
                tamper_sensor = self.sensors.get("tamper0")
                value = 1 if tamper_sensor and tamper_sensor["value"] > 0 else 0
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

    # Sensor Array specific operations

    def list_sensors(self) -> OperationResult:
        """List all sensors in the array"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        sensors = []
        for sensor_id, sensor_info in self.sensors.items():
            sensors.append({
                "id": sensor_id,
                "name": sensor_info["name"],
                "type": self._get_sensor_type_name(sensor_info["type"]),
                "status": self._get_sensor_status_name(sensor_info["status"]),
                "value": sensor_info["value"],
                "unit": sensor_info["unit"],
                "alert_level": self._get_alert_level_name(sensor_info.get("alert_level", AlertLevel.NORMAL)),
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "sensors": sensors,
            "total": len(sensors),
            "online": sum(1 for s in sensors if s["status"] == "Online"),
        })

    def get_sensor_info(self, sensor_id: str) -> OperationResult:
        """Get detailed sensor information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if sensor_id not in self.sensors:
            return OperationResult(False, error=f"Sensor {sensor_id} not found")

        sensor = self.sensors[sensor_id]

        self._record_operation(True)
        return OperationResult(True, data={
            "sensor_id": sensor_id,
            "name": sensor["name"],
            "type": self._get_sensor_type_name(sensor["type"]),
            "status": self._get_sensor_status_name(sensor["status"]),
            "value": sensor["value"],
            "unit": sensor["unit"],
            "min_value": sensor.get("min_value", 0),
            "max_value": sensor.get("max_value", 100),
            "threshold_low": sensor.get("threshold_low", 0),
            "threshold_high": sensor.get("threshold_high", 100),
            "alert_level": self._get_alert_level_name(sensor.get("alert_level", AlertLevel.NORMAL)),
            "last_calibration": sensor.get("last_calibration", "2025-01-01T00:00:00Z"),
        })

    def get_environmental_summary(self) -> OperationResult:
        """Get environmental sensor summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_all_sensors()

        summary = {
            "temperature": {
                "celsius": self.sensors["temp0"]["value"],
                "fahrenheit": self.sensors["temp0"]["value"] * 9/5 + 32,
                "status": "normal" if 15 <= self.sensors["temp0"]["value"] <= 30 else "warning",
            },
            "humidity": {
                "percent": self.sensors["humid0"]["value"],
                "status": "normal" if 30 <= self.sensors["humid0"]["value"] <= 70 else "warning",
            },
            "pressure": {
                "hpa": self.sensors["press0"]["value"],
                "status": "normal",
            },
            "light": {
                "lux": self.sensors["light0"]["value"],
                "condition": self._get_light_condition(self.sensors["light0"]["value"]),
            },
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_security_summary(self) -> OperationResult:
        """Get security sensor summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_all_sensors()

        summary = {
            "motion_detected": self.sensors["motion0"]["value"] > 10,
            "motion_level": self.sensors["motion0"]["value"],
            "tamper_detected": self.sensors["tamper0"]["value"] > 0,
            "vibration_detected": self.sensors["vib0"]["value"] > 20,
            "vibration_level": self.sensors["vib0"]["value"],
            "overall_status": "secure",
        }

        # Determine overall status
        if summary["tamper_detected"]:
            summary["overall_status"] = "tamper_alert"
        elif summary["motion_detected"]:
            summary["overall_status"] = "motion_detected"
        elif summary["vibration_detected"]:
            summary["overall_status"] = "vibration_detected"

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_radiation_status(self) -> OperationResult:
        """Get radiation sensor status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_all_sensors()

        radiation_level = self.sensors["rad0"]["value"]

        # Radiation safety thresholds (μSv/h)
        status = "safe"
        if radiation_level > 1.0:
            status = "elevated"
        if radiation_level > 5.0:
            status = "warning"
        if radiation_level > 10.0:
            status = "danger"

        self._record_operation(True)
        return OperationResult(True, data={
            "level_usv_per_hour": radiation_level,
            "status": status,
            "safe_threshold": 1.0,
            "warning_threshold": 5.0,
            "danger_threshold": 10.0,
        })

    def get_fusion_data(self) -> OperationResult:
        """Get sensor fusion data"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if not self.fusion_enabled:
            return OperationResult(False, error="Sensor fusion not enabled")

        self._update_all_sensors()

        # Compute fused situational awareness
        fusion_data = {
            "situational_awareness": {
                "environment_stable": self._assess_environment_stability(),
                "security_status": self._assess_security_status(),
                "threat_level": self._assess_threat_level(),
                "confidence": 0.85,
            },
            "sensor_count": len(self.sensors),
            "online_sensors": sum(1 for s in self.sensors.values()
                                if s["status"] == SensorStatus.ONLINE),
            "fusion_enabled": self.fusion_enabled,
        }

        self._record_operation(True)
        return OperationResult(True, data=fusion_data)

    def get_alert_summary(self) -> OperationResult:
        """Get alert summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        alerts = []
        for sensor_id, sensor in self.sensors.items():
            alert_level = sensor.get("alert_level", AlertLevel.NORMAL)
            if alert_level > AlertLevel.NORMAL:
                alerts.append({
                    "sensor": sensor["name"],
                    "level": self._get_alert_level_name(alert_level),
                    "value": sensor["value"],
                    "unit": sensor["unit"],
                })

        self._record_operation(True)
        return OperationResult(True, data={
            "alerts": alerts,
            "total": len(alerts),
            "highest_level": self._get_alert_level_name(self._get_highest_alert_level()),
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor array statistics"""
        stats = super().get_statistics()

        active_sensors = sum(1 for s in self.sensors.values()
                           if s["status"] == SensorStatus.ONLINE)

        stats.update({
            "total_sensors": len(self.sensors),
            "active_sensors": active_sensors,
            "uptime_percent": (active_sensors / len(self.sensors) * 100) if self.sensors else 0,
            "fusion_enabled": self.fusion_enabled,
            "history_samples": len(self.sensor_history),
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read sensor array status register (simulated)"""
        status = self.STATUS_ARRAY_ACTIVE | self.STATUS_LOGGING_ENABLED

        active_sensors = sum(1 for s in self.sensors.values()
                           if s["status"] == SensorStatus.ONLINE)

        if active_sensors == len(self.sensors):
            status |= self.STATUS_CALIBRATED
            status |= self.STATUS_ALL_SENSORS_OK

        if self.fusion_enabled:
            status |= self.STATUS_FUSION_ENABLED

        if self._get_highest_alert_level() >= AlertLevel.WARNING:
            status |= self.STATUS_ALERT_ACTIVE

        if self.sensors.get("tamper0", {}).get("value", 0) > 0:
            status |= self.STATUS_TAMPER_DETECTED

        if self.sensors.get("motion0", {}).get("value", 0) > 10:
            status |= self.STATUS_MOTION_DETECTED

        return status

    def _initialize_sensors(self):
        """Initialize all sensors"""
        self.sensors = {
            "temp0": {
                "name": "Temperature Sensor",
                "type": SensorType.TEMPERATURE,
                "status": SensorStatus.ONLINE,
                "value": 22.5,
                "unit": "°C",
                "threshold_low": 0,
                "threshold_high": 50,
                "alert_level": AlertLevel.NORMAL,
            },
            "humid0": {
                "name": "Humidity Sensor",
                "type": SensorType.HUMIDITY,
                "status": SensorStatus.ONLINE,
                "value": 45.0,
                "unit": "%",
                "threshold_low": 20,
                "threshold_high": 80,
                "alert_level": AlertLevel.NORMAL,
            },
            "press0": {
                "name": "Pressure Sensor",
                "type": SensorType.PRESSURE,
                "status": SensorStatus.ONLINE,
                "value": 1013.25,
                "unit": "hPa",
                "alert_level": AlertLevel.NORMAL,
            },
            "motion0": {
                "name": "Motion Detector",
                "type": SensorType.MOTION,
                "status": SensorStatus.ONLINE,
                "value": 0,
                "unit": "level",
                "threshold_high": 50,
                "alert_level": AlertLevel.NORMAL,
            },
            "vib0": {
                "name": "Vibration Sensor",
                "type": SensorType.VIBRATION,
                "status": SensorStatus.ONLINE,
                "value": 5,
                "unit": "level",
                "threshold_high": 50,
                "alert_level": AlertLevel.NORMAL,
            },
            "light0": {
                "name": "Light Sensor",
                "type": SensorType.LIGHT,
                "status": SensorStatus.ONLINE,
                "value": 350,
                "unit": "lux",
                "alert_level": AlertLevel.NORMAL,
            },
            "acoustic0": {
                "name": "Acoustic Sensor",
                "type": SensorType.ACOUSTIC,
                "status": SensorStatus.ONLINE,
                "value": 40,
                "unit": "dB",
                "threshold_high": 85,
                "alert_level": AlertLevel.NORMAL,
            },
            "rad0": {
                "name": "Radiation Detector",
                "type": SensorType.RADIATION,
                "status": SensorStatus.ONLINE,
                "value": 0.12,
                "unit": "μSv/h",
                "threshold_high": 1.0,
                "alert_level": AlertLevel.NORMAL,
            },
            "tamper0": {
                "name": "Tamper Detector",
                "type": SensorType.TAMPER,
                "status": SensorStatus.ONLINE,
                "value": 0,
                "unit": "state",
                "alert_level": AlertLevel.NORMAL,
            },
        }

    def _update_all_sensors(self):
        """Update all sensor readings (simulated)"""
        # Update with simulated realistic values
        self.sensors["temp0"]["value"] = 20.0 + random.random() * 10.0
        self.sensors["humid0"]["value"] = 40.0 + random.random() * 20.0
        self.sensors["press0"]["value"] = 1013.0 + random.random() * 5.0
        self.sensors["motion0"]["value"] = int(random.random() * 10)
        self.sensors["vib0"]["value"] = int(random.random() * 15)
        self.sensors["light0"]["value"] = 300 + random.random() * 200
        self.sensors["acoustic0"]["value"] = 35 + random.random() * 15
        self.sensors["rad0"]["value"] = 0.10 + random.random() * 0.05

        # Store in history
        snapshot = {sensor_id: sensor["value"] for sensor_id, sensor in self.sensors.items()}
        snapshot["timestamp"] = time.time()
        self.sensor_history.append(snapshot)

        if len(self.sensor_history) > self.max_history:
            self.sensor_history = self.sensor_history[-self.max_history:]

    def _get_highest_alert_level(self) -> int:
        """Get highest alert level across all sensors"""
        return max((s.get("alert_level", AlertLevel.NORMAL) for s in self.sensors.values()),
                  default=AlertLevel.NORMAL)

    def _assess_environment_stability(self) -> bool:
        """Assess if environment is stable"""
        temp_ok = 15 <= self.sensors["temp0"]["value"] <= 30
        humid_ok = 30 <= self.sensors["humid0"]["value"] <= 70
        return temp_ok and humid_ok

    def _assess_security_status(self) -> str:
        """Assess security status"""
        if self.sensors["tamper0"]["value"] > 0:
            return "compromised"
        if self.sensors["motion0"]["value"] > 20:
            return "alert"
        return "secure"

    def _assess_threat_level(self) -> str:
        """Assess overall threat level"""
        if self.sensors["rad0"]["value"] > 1.0:
            return "elevated"
        if self.sensors["tamper0"]["value"] > 0:
            return "high"
        if self.sensors["motion0"]["value"] > 30:
            return "medium"
        return "low"

    def _get_light_condition(self, lux: float) -> str:
        """Get light condition description"""
        if lux < 10:
            return "dark"
        elif lux < 100:
            return "dim"
        elif lux < 500:
            return "indoor"
        elif lux < 1000:
            return "bright"
        else:
            return "very_bright"

    def _get_sensor_type_name(self, sensor_type: int) -> str:
        """Get sensor type name"""
        names = {
            SensorType.TEMPERATURE: "Temperature",
            SensorType.HUMIDITY: "Humidity",
            SensorType.PRESSURE: "Pressure",
            SensorType.MOTION: "Motion",
            SensorType.VIBRATION: "Vibration",
            SensorType.LIGHT: "Light",
            SensorType.ACOUSTIC: "Acoustic",
            SensorType.RADIATION: "Radiation",
            SensorType.CHEMICAL: "Chemical",
            SensorType.BIOLOGICAL: "Biological",
            SensorType.MAGNETIC: "Magnetic",
            SensorType.TAMPER: "Tamper",
        }
        return names.get(sensor_type, "Unknown")

    def _get_sensor_status_name(self, status: int) -> str:
        """Get sensor status name"""
        names = {
            SensorStatus.OFFLINE: "Offline",
            SensorStatus.INITIALIZING: "Initializing",
            SensorStatus.ONLINE: "Online",
            SensorStatus.CALIBRATING: "Calibrating",
            SensorStatus.ERROR: "Error",
            SensorStatus.DISABLED: "Disabled",
        }
        return names.get(status, "Unknown")

    def _get_alert_level_name(self, level: int) -> str:
        """Get alert level name"""
        names = {
            AlertLevel.NORMAL: "Normal",
            AlertLevel.INFO: "Info",
            AlertLevel.WARNING: "Warning",
            AlertLevel.ALERT: "Alert",
            AlertLevel.CRITICAL: "Critical",
        }
        return names.get(level, "Unknown")

#!/usr/bin/env python3
"""
Device 0x801E: Tactical Display Control

Military-grade display management with security zones, content protection,
and TEMPEST compliance for secure visual information handling.

Device ID: 0x801E
Group: 2 (Network/Communications)
Risk Level: MONITORED (Display configuration changes logged)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class SecurityZone:
    """Display security zone levels"""
    PUBLIC = 0
    SENSITIVE = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4


class DisplayMode:
    """Tactical display modes"""
    STANDARD = 0
    SECURE = 1
    COVERT = 2
    NIGHT_VISION = 3
    TACTICAL_OVERLAY = 4


class ContentProtection:
    """Content protection levels"""
    NONE = 0
    WATERMARK = 1
    CAPTURE_BLOCK = 2
    FULL_PROTECTION = 3


class TacticalDisplayDevice(DSMILDeviceBase):
    """Tactical Display Control (0x801E)"""

    # Register map
    REG_DISPLAY_STATUS = 0x00
    REG_SECURITY_ZONE = 0x04
    REG_DISPLAY_MODE = 0x08
    REG_PROTECTION_LEVEL = 0x0C
    REG_BRIGHTNESS = 0x10
    REG_TEMPEST_STATUS = 0x14
    REG_ACTIVE_DISPLAYS = 0x18
    REG_OVERLAY_STATUS = 0x1C

    # Status bits
    STATUS_DISPLAY_ACTIVE = 0x01
    STATUS_SECURE_MODE = 0x02
    STATUS_PROTECTION_ENABLED = 0x04
    STATUS_TEMPEST_COMPLIANT = 0x08
    STATUS_OVERLAY_ACTIVE = 0x10
    STATUS_NIGHT_MODE = 0x20
    STATUS_WATERMARK_ACTIVE = 0x40
    STATUS_CAPTURE_BLOCKED = 0x80

    def __init__(self, device_id: int = 0x801E,
                 name: str = "Tactical Display Control",
                 description: str = "Military-Grade Display Security and Management"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.displays = {}
        self.security_zone = SecurityZone.PUBLIC
        self.display_mode = DisplayMode.STANDARD
        self.protection_level = ContentProtection.NONE
        self.tempest_compliant = True
        self.overlay_enabled = False

        # Initialize default display
        self._initialize_default_display()

        # Register map
        self.register_map = {
            "DISPLAY_STATUS": {
                "offset": self.REG_DISPLAY_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Display controller status"
            },
            "SECURITY_ZONE": {
                "offset": self.REG_SECURITY_ZONE,
                "size": 4,
                "access": "RW",
                "description": "Current security zone level"
            },
            "DISPLAY_MODE": {
                "offset": self.REG_DISPLAY_MODE,
                "size": 4,
                "access": "RW",
                "description": "Active display mode"
            },
            "PROTECTION_LEVEL": {
                "offset": self.REG_PROTECTION_LEVEL,
                "size": 4,
                "access": "RW",
                "description": "Content protection level"
            },
            "BRIGHTNESS": {
                "offset": self.REG_BRIGHTNESS,
                "size": 4,
                "access": "RW",
                "description": "Display brightness (0-100)"
            },
            "TEMPEST_STATUS": {
                "offset": self.REG_TEMPEST_STATUS,
                "size": 4,
                "access": "RO",
                "description": "TEMPEST compliance status"
            },
            "ACTIVE_DISPLAYS": {
                "offset": self.REG_ACTIVE_DISPLAYS,
                "size": 4,
                "access": "RO",
                "description": "Number of active displays"
            },
            "OVERLAY_STATUS": {
                "offset": self.REG_OVERLAY_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Tactical overlay status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Tactical Display Control device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize with secure defaults
            self.security_zone = SecurityZone.PUBLIC
            self.display_mode = DisplayMode.STANDARD
            self.protection_level = ContentProtection.WATERMARK
            self.tempest_compliant = True
            self._initialize_default_display()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "displays": len(self.displays),
                "security_zone": self._get_zone_name(self.security_zone),
                "display_mode": self._get_mode_name(self.display_mode),
                "protection": self._get_protection_name(self.protection_level),
                "tempest_compliant": self.tempest_compliant,
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_WRITE,
            DeviceCapability.CONFIGURATION,
            DeviceCapability.STATUS_REPORTING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "display_active": bool(status_reg & self.STATUS_DISPLAY_ACTIVE),
            "secure_mode": bool(status_reg & self.STATUS_SECURE_MODE),
            "protection_enabled": bool(status_reg & self.STATUS_PROTECTION_ENABLED),
            "tempest_compliant": bool(status_reg & self.STATUS_TEMPEST_COMPLIANT),
            "overlay_active": bool(status_reg & self.STATUS_OVERLAY_ACTIVE),
            "night_mode": bool(status_reg & self.STATUS_NIGHT_MODE),
            "watermark_active": bool(status_reg & self.STATUS_WATERMARK_ACTIVE),
            "capture_blocked": bool(status_reg & self.STATUS_CAPTURE_BLOCKED),
            "active_displays": len(self.displays),
            "security_zone": self._get_zone_name(self.security_zone),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "DISPLAY_STATUS":
                value = self._read_status_register()
            elif register == "SECURITY_ZONE":
                value = self.security_zone
            elif register == "DISPLAY_MODE":
                value = self.display_mode
            elif register == "PROTECTION_LEVEL":
                value = self.protection_level
            elif register == "BRIGHTNESS":
                value = self.displays.get("primary", {}).get("brightness", 50)
            elif register == "TEMPEST_STATUS":
                value = 0x01 if self.tempest_compliant else 0x00
            elif register == "ACTIVE_DISPLAYS":
                value = len(self.displays)
            elif register == "OVERLAY_STATUS":
                value = 0x01 if self.overlay_enabled else 0x00
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

    # Tactical Display specific operations

    def list_displays(self) -> OperationResult:
        """List all connected displays"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        displays = []
        for display_id, display_info in self.displays.items():
            displays.append({
                "id": display_id,
                "name": display_info["name"],
                "resolution": display_info["resolution"],
                "active": display_info["active"],
                "brightness": display_info["brightness"],
                "security_zone": self._get_zone_name(display_info.get("zone", SecurityZone.PUBLIC)),
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "displays": displays,
            "total": len(displays),
        })

    def get_display_info(self, display_id: str) -> OperationResult:
        """Get detailed display information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if display_id not in self.displays:
            return OperationResult(False, error=f"Display {display_id} not found")

        display = self.displays[display_id]

        self._record_operation(True)
        return OperationResult(True, data={
            "display_id": display_id,
            "name": display["name"],
            "resolution": display["resolution"],
            "active": display["active"],
            "brightness": display["brightness"],
            "security_zone": self._get_zone_name(display.get("zone", SecurityZone.PUBLIC)),
            "mode": self._get_mode_name(self.display_mode),
            "protection": self._get_protection_name(self.protection_level),
        })

    def get_security_config(self) -> OperationResult:
        """Get security configuration"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        config = {
            "security_zone": self._get_zone_name(self.security_zone),
            "protection_level": self._get_protection_name(self.protection_level),
            "tempest_compliant": self.tempest_compliant,
            "capture_blocking": self.protection_level >= ContentProtection.CAPTURE_BLOCK,
            "watermarking": self.protection_level >= ContentProtection.WATERMARK,
            "encryption": self.display_mode == DisplayMode.SECURE,
        }

        self._record_operation(True)
        return OperationResult(True, data=config)

    def get_display_modes(self) -> OperationResult:
        """Get available display modes"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        modes = [
            {
                "mode": "STANDARD",
                "id": DisplayMode.STANDARD,
                "description": "Standard display mode",
                "active": self.display_mode == DisplayMode.STANDARD,
            },
            {
                "mode": "SECURE",
                "id": DisplayMode.SECURE,
                "description": "Encrypted display with full protection",
                "active": self.display_mode == DisplayMode.SECURE,
            },
            {
                "mode": "COVERT",
                "id": DisplayMode.COVERT,
                "description": "Low-visibility covert operations mode",
                "active": self.display_mode == DisplayMode.COVERT,
            },
            {
                "mode": "NIGHT_VISION",
                "id": DisplayMode.NIGHT_VISION,
                "description": "Night vision compatible display",
                "active": self.display_mode == DisplayMode.NIGHT_VISION,
            },
            {
                "mode": "TACTICAL_OVERLAY",
                "id": DisplayMode.TACTICAL_OVERLAY,
                "description": "Tactical information overlay",
                "active": self.display_mode == DisplayMode.TACTICAL_OVERLAY,
            },
        ]

        self._record_operation(True)
        return OperationResult(True, data={
            "modes": modes,
            "current_mode": self._get_mode_name(self.display_mode),
        })

    def get_tempest_status(self) -> OperationResult:
        """Get TEMPEST compliance status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Simulated TEMPEST measurements
        tempest_info = {
            "compliant": self.tempest_compliant,
            "emission_level": "LOW" if self.tempest_compliant else "MODERATE",
            "shielding_active": self.display_mode in [DisplayMode.SECURE, DisplayMode.COVERT],
            "last_test": "2025-01-05T12:00:00Z",
            "certification": "NSA/CSS EPL" if self.tempest_compliant else None,
        }

        self._record_operation(True)
        return OperationResult(True, data=tempest_info)

    def get_overlay_status(self) -> OperationResult:
        """Get tactical overlay status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        overlay_info = {
            "enabled": self.overlay_enabled,
            "layers": [],
        }

        if self.overlay_enabled:
            overlay_info["layers"] = [
                {"name": "grid", "active": True},
                {"name": "coordinates", "active": True},
                {"name": "threat_indicators", "active": False},
                {"name": "nav_waypoints", "active": True},
            ]

        self._record_operation(True)
        return OperationResult(True, data=overlay_info)

    def get_protection_features(self) -> OperationResult:
        """Get content protection features"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        features = {
            "screenshot_blocking": self.protection_level >= ContentProtection.CAPTURE_BLOCK,
            "screen_recording_blocking": self.protection_level >= ContentProtection.CAPTURE_BLOCK,
            "watermarking": self.protection_level >= ContentProtection.WATERMARK,
            "hdcp_enforcement": self.protection_level >= ContentProtection.FULL_PROTECTION,
            "secure_composition": self.display_mode == DisplayMode.SECURE,
            "anti_reflection": self.display_mode in [DisplayMode.SECURE, DisplayMode.COVERT],
        }

        self._record_operation(True)
        return OperationResult(True, data=features)

    def get_statistics(self) -> Dict[str, Any]:
        """Get display statistics"""
        stats = super().get_statistics()

        stats.update({
            "total_displays": len(self.displays),
            "active_displays": sum(1 for d in self.displays.values() if d["active"]),
            "security_zone": self._get_zone_name(self.security_zone),
            "display_mode": self._get_mode_name(self.display_mode),
            "protection_level": self._get_protection_name(self.protection_level),
            "tempest_compliant": self.tempest_compliant,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read display status register (simulated)"""
        status = 0

        if any(d["active"] for d in self.displays.values()):
            status |= self.STATUS_DISPLAY_ACTIVE

        if self.display_mode in [DisplayMode.SECURE, DisplayMode.COVERT]:
            status |= self.STATUS_SECURE_MODE

        if self.protection_level >= ContentProtection.WATERMARK:
            status |= self.STATUS_PROTECTION_ENABLED

        if self.tempest_compliant:
            status |= self.STATUS_TEMPEST_COMPLIANT

        if self.overlay_enabled:
            status |= self.STATUS_OVERLAY_ACTIVE

        if self.display_mode == DisplayMode.NIGHT_VISION:
            status |= self.STATUS_NIGHT_MODE

        if self.protection_level >= ContentProtection.WATERMARK:
            status |= self.STATUS_WATERMARK_ACTIVE

        if self.protection_level >= ContentProtection.CAPTURE_BLOCK:
            status |= self.STATUS_CAPTURE_BLOCKED

        return status

    def _initialize_default_display(self):
        """Initialize default display"""
        self.displays = {
            "primary": {
                "name": "Primary Display",
                "resolution": "1920x1080",
                "active": True,
                "brightness": 70,
                "zone": SecurityZone.PUBLIC,
            }
        }

    def _get_zone_name(self, zone: int) -> str:
        """Get security zone name"""
        names = {
            SecurityZone.PUBLIC: "Public",
            SecurityZone.SENSITIVE: "Sensitive",
            SecurityZone.CONFIDENTIAL: "Confidential",
            SecurityZone.SECRET: "Secret",
            SecurityZone.TOP_SECRET: "Top Secret",
        }
        return names.get(zone, "Unknown")

    def _get_mode_name(self, mode: int) -> str:
        """Get display mode name"""
        names = {
            DisplayMode.STANDARD: "Standard",
            DisplayMode.SECURE: "Secure",
            DisplayMode.COVERT: "Covert",
            DisplayMode.NIGHT_VISION: "Night Vision",
            DisplayMode.TACTICAL_OVERLAY: "Tactical Overlay",
        }
        return names.get(mode, "Unknown")

    def _get_protection_name(self, protection: int) -> str:
        """Get protection level name"""
        names = {
            ContentProtection.NONE: "None",
            ContentProtection.WATERMARK: "Watermark",
            ContentProtection.CAPTURE_BLOCK: "Capture Block",
            ContentProtection.FULL_PROTECTION: "Full Protection",
        }
        return names.get(protection, "Unknown")

#!/usr/bin/env python3
"""
DSMIL Security Monitor
Real-time security monitoring and threat detection
"""

import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityThreat:
    """Security threat information"""
    threat_id: str
    threat_type: str
    severity: str
    description: str
    detected_at: datetime
    source: str
    indicators: List[str]
    mitigated: bool = False


class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self):
        self.active_threats: List[SecurityThreat] = []
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "temperature": 0.0,
            "network_connections": 0
        }
        self.security_status = {
            "threat_level": "LOW",
            "active_alerts": 0,
            "auth_failures": 0,
            "last_scan": None
        }
        
    async def initialize(self):
        """Initialize security monitor"""
        logger.info("Initializing security monitor...")
        await self._update_system_metrics()
        logger.info("Security monitor initialized")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._update_system_metrics()
                await self._check_security_threats()
                await self._update_security_status()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Security monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            self.system_metrics.update({
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_connections": len(psutil.net_connections()),
                "timestamp": datetime.utcnow()
            })
            
            # Get temperature if available
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    for name, entries in temps.items():
                        if 'coretemp' in name.lower() or 'cpu' in name.lower():
                            if entries:
                                self.system_metrics["temperature"] = entries[0].current
                                break
            except:
                pass  # Temperature monitoring not available
                
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    async def _check_security_threats(self):
        """Check for security threats"""
        try:
            # Check CPU usage anomalies
            if self.system_metrics["cpu_usage"] > 90:
                await self._log_security_event(
                    "HIGH_CPU_USAGE",
                    "HIGH",
                    f"CPU usage at {self.system_metrics['cpu_usage']:.1f}%",
                    ["cpu_anomaly", "performance_degradation"]
                )
            
            # Check memory usage
            if self.system_metrics["memory_usage"] > 85:
                await self._log_security_event(
                    "HIGH_MEMORY_USAGE", 
                    "MODERATE",
                    f"Memory usage at {self.system_metrics['memory_usage']:.1f}%",
                    ["memory_anomaly"]
                )
            
            # Check disk usage
            if self.system_metrics["disk_usage"] > 90:
                await self._log_security_event(
                    "LOW_DISK_SPACE",
                    "HIGH", 
                    f"Disk usage at {self.system_metrics['disk_usage']:.1f}%",
                    ["disk_full", "availability_risk"]
                )
            
            # Check temperature
            if self.system_metrics.get("temperature", 0) > 85:
                await self._log_security_event(
                    "HIGH_TEMPERATURE",
                    "CRITICAL",
                    f"System temperature at {self.system_metrics['temperature']:.1f}°C", 
                    ["thermal_throttling", "hardware_stress"]
                )
                
        except Exception as e:
            logger.error(f"Security threat check failed: {e}")
    
    async def _log_security_event(self, event_type: str, severity: str, description: str, indicators: List[str]):
        """Log security event"""
        # Check if this is a duplicate recent event
        recent_threshold = datetime.utcnow() - timedelta(minutes=5)
        
        for threat in self.active_threats:
            if (threat.threat_type == event_type and 
                threat.detected_at > recent_threshold and 
                not threat.mitigated):
                return  # Don't log duplicate recent events
        
        threat = SecurityThreat(
            threat_id=f"{event_type}_{int(datetime.utcnow().timestamp())}",
            threat_type=event_type,
            severity=severity,
            description=description,
            detected_at=datetime.utcnow(),
            source="system_monitor",
            indicators=indicators
        )
        
        self.active_threats.append(threat)
        
        # Keep only last 100 threats
        if len(self.active_threats) > 100:
            self.active_threats = self.active_threats[-100:]
        
        logger.warning(f"Security event: {event_type} - {description}")
    
    async def _update_security_status(self):
        """Update overall security status"""
        active_threats = [t for t in self.active_threats if not t.mitigated]
        
        # Determine threat level
        critical_threats = [t for t in active_threats if t.severity == "CRITICAL"]
        high_threats = [t for t in active_threats if t.severity == "HIGH"] 
        moderate_threats = [t for t in active_threats if t.severity == "MODERATE"]
        
        if critical_threats:
            threat_level = "CRITICAL"
        elif high_threats:
            threat_level = "HIGH"
        elif moderate_threats:
            threat_level = "MODERATE"
        else:
            threat_level = "LOW"
        
        self.security_status.update({
            "threat_level": threat_level,
            "active_alerts": len(active_threats),
            "last_scan": datetime.utcnow(),
            "system_health": self._assess_system_health()
        })
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        cpu = self.system_metrics["cpu_usage"]
        memory = self.system_metrics["memory_usage"] 
        disk = self.system_metrics["disk_usage"]
        temp = self.system_metrics.get("temperature", 0)
        
        if cpu > 90 or memory > 90 or disk > 95 or temp > 90:
            return "CRITICAL"
        elif cpu > 80 or memory > 80 or disk > 85 or temp > 80:
            return "WARNING"
        else:
            return "NORMAL"
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            "overall_status": self._assess_system_health(),
            "cpu_usage": self.system_metrics["cpu_usage"],
            "memory_usage": self.system_metrics["memory_usage"],
            "disk_usage": self.system_metrics["disk_usage"],
            "temperature": self.system_metrics.get("temperature", 0),
            "network_connections": self.system_metrics["network_connections"],
            "last_update": self.system_metrics.get("timestamp", datetime.utcnow()).isoformat()
        }
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        active_threats = [t for t in self.active_threats if not t.mitigated]
        
        return {
            "threat_level": self.security_status["threat_level"],
            "active_alerts": len(active_threats),
            "total_threats": len(self.active_threats),
            "mitigated_threats": len([t for t in self.active_threats if t.mitigated]),
            "recent_threats": [
                {
                    "threat_id": t.threat_id,
                    "type": t.threat_type,
                    "severity": t.severity,
                    "description": t.description,
                    "detected_at": t.detected_at.isoformat(),
                    "indicators": t.indicators
                }
                for t in active_threats[-10:]  # Last 10 active threats
            ],
            "last_scan": self.security_status["last_scan"].isoformat() if self.security_status["last_scan"] else None
        }
    
    async def mitigate_threat(self, threat_id: str, mitigation_notes: str = "") -> bool:
        """Mark threat as mitigated"""
        for threat in self.active_threats:
            if threat.threat_id == threat_id:
                threat.mitigated = True
                logger.info(f"Threat {threat_id} mitigated: {mitigation_notes}")
                return True
        
        return False
    
    async def cleanup(self):
        """Cleanup security monitor"""
        logger.info("Security monitor cleanup complete")


# Emergency stop controller
class EmergencyStopController:
    """Emergency stop system controller"""
    
    def __init__(self):
        self.is_active = False
        self.activated_at = None
        self.activated_by = None
        self.reason = None
        
    async def activate(self, reason: str, triggered_by: str):
        """Activate emergency stop"""
        self.is_active = True
        self.activated_at = datetime.utcnow()
        self.activated_by = triggered_by
        self.reason = reason
        
        logger.critical(f"EMERGENCY STOP ACTIVATED by {triggered_by}: {reason}")
        
    async def deactivate(self, deactivated_by: str):
        """Deactivate emergency stop"""
        self.is_active = False
        logger.info(f"Emergency stop deactivated by {deactivated_by}")
        
    async def is_active(self) -> bool:
        """Check if emergency stop is active"""
        return self.is_active
    
    async def get_status(self) -> Dict[str, Any]:
        """Get emergency stop status"""
        return {
            "is_active": self.is_active,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "activated_by": self.activated_by,
            "reason": self.reason
        }


# Global instances
security_monitor = SecurityMonitor()
emergency_stop = EmergencyStopController()
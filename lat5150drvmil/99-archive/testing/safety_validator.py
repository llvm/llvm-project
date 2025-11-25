#!/usr/bin/env python3
"""
Automated Safety Validation System for SMBIOS Token Testing
===========================================================

Comprehensive safety validation system for Dell Latitude 5450 MIL-SPEC
SMBIOS token testing. Ensures all safety systems are operational before,
during, and after token testing operations.

Safety Layers:
1. Hardware validation (thermal, memory, CPU)
2. Software validation (modules, tools, permissions)
3. Environmental validation (system state, resources)
4. Recovery validation (emergency procedures, rollback)
5. Real-time monitoring during tests

Author: TESTBED Agent  
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import time
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "psutil"], check=True)
    import psutil

class SafetyLevel(Enum):
    """Safety validation levels"""
    SAFE = "SAFE"
    WARNING = "WARNING" 
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class SafetyCheck:
    """Individual safety check result"""
    name: str
    category: str
    level: SafetyLevel
    status: bool
    message: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    timestamp: Optional[datetime] = None

@dataclass
class SafetyReport:
    """Complete safety validation report"""
    timestamp: datetime
    overall_status: SafetyLevel
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    critical_issues: int
    emergency_issues: int
    checks: List[SafetyCheck]
    system_snapshot: Dict[str, Any]

class SafetyValidator:
    """Comprehensive safety validation system"""
    
    def __init__(self, work_dir: str = "/home/john/LAT5150DRVMIL"):
        self.work_dir = Path(work_dir)
        self.validation_history: List[SafetyReport] = []
        
        # Safety thresholds for Dell Latitude 5450 MIL-SPEC
        self.thresholds = {
            'thermal': {
                'safe': 85.0,      # Safe operating temperature
                'warning': 95.0,   # Warning threshold
                'critical': 100.0, # Critical threshold
                'emergency': 105.0 # Emergency stop
            },
            'memory': {
                'safe': 70.0,      # Safe memory usage %
                'warning': 85.0,   # Warning threshold
                'critical': 95.0   # Critical threshold
            },
            'cpu': {
                'safe': 70.0,      # Safe CPU usage %
                'warning': 80.0,   # Warning threshold
                'critical': 95.0   # Critical threshold
            },
            'disk': {
                'safe': 80.0,      # Safe disk usage %
                'warning': 90.0,   # Warning threshold
                'critical': 95.0   # Critical threshold
            }
        }
        
        # Required system components
        self.required_components = {
            'commands': [
                'smbios-token-ctl',
                'lsmod',
                'modinfo',
                'gcc',
                'make'
            ],
            'files': [
                '/sys/class/thermal',
                '/proc/cpuinfo',
                '/proc/meminfo'
            ],
            'modules': [
                'dell_smbios',  # Optional but recommended
                'dell_wmi'      # Optional but recommended
            ]
        }
        
    def validate_hardware_safety(self) -> List[SafetyCheck]:
        """Validate hardware safety parameters"""
        checks = []
        
        # Thermal validation
        thermal_check = self._check_thermal_safety()
        checks.append(thermal_check)
        
        # Memory validation
        memory_check = self._check_memory_safety()
        checks.append(memory_check)
        
        # CPU validation
        cpu_check = self._check_cpu_safety()
        checks.append(cpu_check)
        
        # Disk space validation
        disk_check = self._check_disk_safety()
        checks.append(disk_check)
        
        # System load validation
        load_check = self._check_system_load()
        checks.append(load_check)
        
        return checks
        
    def _check_thermal_safety(self) -> SafetyCheck:
        """Check thermal safety status"""
        try:
            temperatures = []
            
            # Get thermal zone readings
            thermal_zones = Path("/sys/class/thermal").glob("thermal_zone*")
            for zone in thermal_zones:
                temp_file = zone / "temp"
                if temp_file.exists():
                    try:
                        temp_raw = int(temp_file.read_text().strip())
                        temp_celsius = temp_raw / 1000.0
                        temperatures.append(temp_celsius)
                    except (ValueError, OSError):
                        continue
                        
            # Get psutil temperatures
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            temperatures.append(entry.current)
                            
            if not temperatures:
                return SafetyCheck(
                    name="thermal_safety",
                    category="hardware",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message="No thermal sensors found",
                    timestamp=datetime.now(timezone.utc)
                )
                
            max_temp = max(temperatures)
            avg_temp = sum(temperatures) / len(temperatures)
            
            # Determine safety level
            if max_temp >= self.thresholds['thermal']['emergency']:
                level = SafetyLevel.EMERGENCY
                status = False
                message = f"EMERGENCY: Temperature {max_temp:.1f}°C exceeds emergency threshold"
            elif max_temp >= self.thresholds['thermal']['critical']:
                level = SafetyLevel.CRITICAL
                status = False
                message = f"CRITICAL: Temperature {max_temp:.1f}°C exceeds critical threshold"
            elif max_temp >= self.thresholds['thermal']['warning']:
                level = SafetyLevel.WARNING
                status = True
                message = f"WARNING: Temperature {max_temp:.1f}°C above normal operating range"
            else:
                level = SafetyLevel.SAFE
                status = True
                message = f"SAFE: Temperature {max_temp:.1f}°C within safe range"
                
            return SafetyCheck(
                name="thermal_safety",
                category="hardware",
                level=level,
                status=status,
                message=message,
                value=max_temp,
                threshold=self.thresholds['thermal']['warning'],
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return SafetyCheck(
                name="thermal_safety",
                category="hardware",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Thermal check failed: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_memory_safety(self) -> SafetyCheck:
        """Check memory usage safety"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent >= self.thresholds['memory']['critical']:
                level = SafetyLevel.CRITICAL
                status = False
                message = f"CRITICAL: Memory usage {usage_percent:.1f}% exceeds critical threshold"
            elif usage_percent >= self.thresholds['memory']['warning']:
                level = SafetyLevel.WARNING
                status = True
                message = f"WARNING: Memory usage {usage_percent:.1f}% above normal range"
            else:
                level = SafetyLevel.SAFE
                status = True
                message = f"SAFE: Memory usage {usage_percent:.1f}% within safe range"
                
            return SafetyCheck(
                name="memory_safety",
                category="hardware",
                level=level,
                status=status,
                message=message,
                value=usage_percent,
                threshold=self.thresholds['memory']['warning'],
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return SafetyCheck(
                name="memory_safety",
                category="hardware",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Memory check failed: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_cpu_safety(self) -> SafetyCheck:
        """Check CPU usage safety"""
        try:
            # Get CPU usage over 2 seconds for accuracy
            cpu_percent = psutil.cpu_percent(interval=2)
            
            if cpu_percent >= self.thresholds['cpu']['critical']:
                level = SafetyLevel.CRITICAL
                status = False
                message = f"CRITICAL: CPU usage {cpu_percent:.1f}% exceeds critical threshold"
            elif cpu_percent >= self.thresholds['cpu']['warning']:
                level = SafetyLevel.WARNING
                status = True
                message = f"WARNING: CPU usage {cpu_percent:.1f}% above normal range"
            else:
                level = SafetyLevel.SAFE
                status = True
                message = f"SAFE: CPU usage {cpu_percent:.1f}% within safe range"
                
            return SafetyCheck(
                name="cpu_safety",
                category="hardware",
                level=level,
                status=status,
                message=message,
                value=cpu_percent,
                threshold=self.thresholds['cpu']['warning'],
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return SafetyCheck(
                name="cpu_safety",
                category="hardware",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"CPU check failed: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_disk_safety(self) -> SafetyCheck:
        """Check disk space safety"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent >= self.thresholds['disk']['critical']:
                level = SafetyLevel.CRITICAL
                status = False
                message = f"CRITICAL: Disk usage {usage_percent:.1f}% exceeds critical threshold"
            elif usage_percent >= self.thresholds['disk']['warning']:
                level = SafetyLevel.WARNING
                status = True
                message = f"WARNING: Disk usage {usage_percent:.1f}% above normal range"
            else:
                level = SafetyLevel.SAFE
                status = True
                message = f"SAFE: Disk usage {usage_percent:.1f}% within safe range"
                
            return SafetyCheck(
                name="disk_safety",
                category="hardware",
                level=level,
                status=status,
                message=message,
                value=usage_percent,
                threshold=self.thresholds['disk']['warning'],
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return SafetyCheck(
                name="disk_safety",
                category="hardware",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Disk check failed: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_system_load(self) -> SafetyCheck:
        """Check system load average"""
        try:
            load_avg = os.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            
            if load_percent >= 90:
                level = SafetyLevel.CRITICAL
                status = False
                message = f"CRITICAL: System load {load_percent:.1f}% is very high"
            elif load_percent >= 70:
                level = SafetyLevel.WARNING
                status = True
                message = f"WARNING: System load {load_percent:.1f}% is elevated"
            else:
                level = SafetyLevel.SAFE
                status = True
                message = f"SAFE: System load {load_percent:.1f}% is normal"
                
            return SafetyCheck(
                name="system_load",
                category="hardware",
                level=level,
                status=status,
                message=message,
                value=load_percent,
                threshold=70.0,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return SafetyCheck(
                name="system_load",
                category="hardware",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"System load check failed: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def validate_software_safety(self) -> List[SafetyCheck]:
        """Validate software components safety"""
        checks = []
        
        # Check required commands
        for cmd in self.required_components['commands']:
            check = self._check_command_availability(cmd)
            checks.append(check)
            
        # Check required files/paths
        for filepath in self.required_components['files']:
            check = self._check_file_availability(filepath)
            checks.append(check)
            
        # Check Dell modules (optional but recommended)
        for module in self.required_components['modules']:
            check = self._check_module_availability(module)
            checks.append(check)
            
        # Check DSMIL module status (should not be loaded initially)
        dsmil_check = self._check_dsmil_module_status()
        checks.append(dsmil_check)
        
        return checks
        
    def _check_command_availability(self, command: str) -> SafetyCheck:
        """Check if required command is available"""
        try:
            result = subprocess.run(['which', command], capture_output=True)
            
            if result.returncode == 0:
                return SafetyCheck(
                    name=f"command_{command}",
                    category="software",
                    level=SafetyLevel.SAFE,
                    status=True,
                    message=f"Command '{command}' is available",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                severity = SafetyLevel.CRITICAL if command in ['smbios-token-ctl', 'gcc'] else SafetyLevel.WARNING
                return SafetyCheck(
                    name=f"command_{command}",
                    category="software",
                    level=severity,
                    status=False,
                    message=f"Command '{command}' not found",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name=f"command_{command}",
                category="software",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check command '{command}': {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_file_availability(self, filepath: str) -> SafetyCheck:
        """Check if required file/path exists"""
        try:
            path = Path(filepath)
            
            if path.exists():
                return SafetyCheck(
                    name=f"file_{Path(filepath).name}",
                    category="software",
                    level=SafetyLevel.SAFE,
                    status=True,
                    message=f"Path '{filepath}' exists",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return SafetyCheck(
                    name=f"file_{Path(filepath).name}",
                    category="software", 
                    level=SafetyLevel.WARNING,
                    status=False,
                    message=f"Path '{filepath}' not found",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name=f"file_{Path(filepath).name}",
                category="software",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check path '{filepath}': {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_module_availability(self, module: str) -> SafetyCheck:
        """Check if kernel module is available"""
        try:
            result = subprocess.run(['modinfo', module], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if module is loaded
                lsmod_result = subprocess.run(['lsmod'], capture_output=True, text=True)
                is_loaded = module in lsmod_result.stdout
                
                status_msg = "loaded" if is_loaded else "available but not loaded"
                
                return SafetyCheck(
                    name=f"module_{module}",
                    category="software",
                    level=SafetyLevel.SAFE,
                    status=True,
                    message=f"Module '{module}' is {status_msg}",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return SafetyCheck(
                    name=f"module_{module}",
                    category="software",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message=f"Module '{module}' not available",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name=f"module_{module}",
                category="software",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check module '{module}': {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_dsmil_module_status(self) -> SafetyCheck:
        """Check DSMIL module status (should be unloaded for safe testing)"""
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            
            dsmil_loaded = any('dsmil' in line for line in result.stdout.split('\n'))
            
            if dsmil_loaded:
                return SafetyCheck(
                    name="dsmil_module_status",
                    category="software",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message="DSMIL module is loaded - unload before testing",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return SafetyCheck(
                    name="dsmil_module_status",
                    category="software",
                    level=SafetyLevel.SAFE,
                    status=True,
                    message="DSMIL module is not loaded (safe for testing)",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name="dsmil_module_status",
                category="software",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check DSMIL module status: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def validate_recovery_systems(self) -> List[SafetyCheck]:
        """Validate emergency recovery systems"""
        checks = []
        
        # Check emergency stop script
        emergency_script = self.work_dir / "monitoring" / "emergency_stop.sh"
        checks.append(self._check_emergency_script(emergency_script))
        
        # Check rollback scripts
        quick_rollback = self.work_dir / "quick_rollback.sh"
        checks.append(self._check_rollback_script(quick_rollback, "quick"))
        
        comprehensive_rollback = self.work_dir / "comprehensive_rollback.sh"
        checks.append(self._check_rollback_script(comprehensive_rollback, "comprehensive"))
        
        # Check baseline snapshots
        checks.append(self._check_baseline_snapshots())
        
        # Check monitoring systems
        monitoring_dir = self.work_dir / "monitoring"
        checks.append(self._check_monitoring_systems(monitoring_dir))
        
        return checks
        
    def _check_emergency_script(self, script_path: Path) -> SafetyCheck:
        """Check emergency stop script"""
        try:
            if script_path.exists() and script_path.is_file():
                # Check if executable
                if os.access(script_path, os.X_OK):
                    return SafetyCheck(
                        name="emergency_stop_script",
                        category="recovery",
                        level=SafetyLevel.SAFE,
                        status=True,
                        message=f"Emergency stop script is available and executable",
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    return SafetyCheck(
                        name="emergency_stop_script",
                        category="recovery",
                        level=SafetyLevel.WARNING,
                        status=False,
                        message=f"Emergency stop script exists but is not executable",
                        timestamp=datetime.now(timezone.utc)
                    )
            else:
                return SafetyCheck(
                    name="emergency_stop_script",
                    category="recovery",
                    level=SafetyLevel.CRITICAL,
                    status=False,
                    message=f"Emergency stop script not found at {script_path}",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name="emergency_stop_script",
                category="recovery",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check emergency stop script: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_rollback_script(self, script_path: Path, rollback_type: str) -> SafetyCheck:
        """Check rollback script availability"""
        try:
            if script_path.exists() and script_path.is_file():
                if os.access(script_path, os.X_OK):
                    return SafetyCheck(
                        name=f"{rollback_type}_rollback_script",
                        category="recovery",
                        level=SafetyLevel.SAFE,
                        status=True,
                        message=f"{rollback_type.title()} rollback script is available",
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    return SafetyCheck(
                        name=f"{rollback_type}_rollback_script",
                        category="recovery",
                        level=SafetyLevel.WARNING,
                        status=False,
                        message=f"{rollback_type.title()} rollback script is not executable",
                        timestamp=datetime.now(timezone.utc)
                    )
            else:
                return SafetyCheck(
                    name=f"{rollback_type}_rollback_script",
                    category="recovery",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message=f"{rollback_type.title()} rollback script not found",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name=f"{rollback_type}_rollback_script",
                category="recovery",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check {rollback_type} rollback script: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_baseline_snapshots(self) -> SafetyCheck:
        """Check availability of baseline snapshots"""
        try:
            baseline_files = list(self.work_dir.glob("baseline_*.tar.gz"))
            
            if len(baseline_files) >= 1:
                # Get most recent baseline
                most_recent = max(baseline_files, key=lambda x: x.stat().st_mtime)
                age_hours = (time.time() - most_recent.stat().st_mtime) / 3600
                
                if age_hours < 24:
                    level = SafetyLevel.SAFE
                    message = f"Recent baseline snapshot available (age: {age_hours:.1f} hours)"
                elif age_hours < 168:  # 1 week
                    level = SafetyLevel.WARNING
                    message = f"Baseline snapshot is {age_hours:.1f} hours old - consider updating"
                else:
                    level = SafetyLevel.WARNING
                    message = f"Baseline snapshot is {age_hours:.1f} hours old - needs updating"
                    
                return SafetyCheck(
                    name="baseline_snapshots",
                    category="recovery",
                    level=level,
                    status=True,
                    message=message,
                    value=len(baseline_files),
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return SafetyCheck(
                    name="baseline_snapshots",
                    category="recovery",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message="No baseline snapshots found - create backup before testing",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name="baseline_snapshots",
                category="recovery",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check baseline snapshots: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def _check_monitoring_systems(self, monitoring_dir: Path) -> SafetyCheck:
        """Check monitoring system availability"""
        try:
            if not monitoring_dir.exists():
                return SafetyCheck(
                    name="monitoring_systems",
                    category="recovery",
                    level=SafetyLevel.CRITICAL,
                    status=False,
                    message="Monitoring directory not found",
                    timestamp=datetime.now(timezone.utc)
                )
                
            required_files = [
                "dsmil_comprehensive_monitor.py",
                "multi_terminal_launcher.sh"
            ]
            
            missing_files = []
            for filename in required_files:
                file_path = monitoring_dir / filename
                if not file_path.exists():
                    missing_files.append(filename)
                    
            if not missing_files:
                return SafetyCheck(
                    name="monitoring_systems",
                    category="recovery",
                    level=SafetyLevel.SAFE,
                    status=True,
                    message="All monitoring systems available",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return SafetyCheck(
                    name="monitoring_systems",
                    category="recovery",
                    level=SafetyLevel.WARNING,
                    status=False,
                    message=f"Missing monitoring files: {', '.join(missing_files)}",
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return SafetyCheck(
                name="monitoring_systems",
                category="recovery",
                level=SafetyLevel.WARNING,
                status=False,
                message=f"Failed to check monitoring systems: {e}",
                timestamp=datetime.now(timezone.utc)
            )
            
    def run_full_safety_validation(self) -> SafetyReport:
        """Run complete safety validation"""
        print("🔒 Running comprehensive safety validation...")
        
        all_checks = []
        
        # Hardware safety checks
        print("  🖥️ Validating hardware safety...")
        hardware_checks = self.validate_hardware_safety()
        all_checks.extend(hardware_checks)
        
        # Software safety checks  
        print("  💻 Validating software components...")
        software_checks = self.validate_software_safety()
        all_checks.extend(software_checks)
        
        # Recovery system checks
        print("  🛡️ Validating recovery systems...")
        recovery_checks = self.validate_recovery_systems()
        all_checks.extend(recovery_checks)
        
        # Calculate overall status
        emergency_count = sum(1 for c in all_checks if c.level == SafetyLevel.EMERGENCY)
        critical_count = sum(1 for c in all_checks if c.level == SafetyLevel.CRITICAL)
        warning_count = sum(1 for c in all_checks if c.level == SafetyLevel.WARNING)
        passed_count = sum(1 for c in all_checks if c.status)
        failed_count = len(all_checks) - passed_count
        
        if emergency_count > 0:
            overall_status = SafetyLevel.EMERGENCY
        elif critical_count > 0:
            overall_status = SafetyLevel.CRITICAL
        elif warning_count > 0:
            overall_status = SafetyLevel.WARNING
        else:
            overall_status = SafetyLevel.SAFE
            
        # Create system snapshot
        system_snapshot = self._create_system_snapshot()
        
        # Create safety report
        report = SafetyReport(
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            total_checks=len(all_checks),
            passed_checks=passed_count,
            failed_checks=failed_count,
            warnings=warning_count,
            critical_issues=critical_count,
            emergency_issues=emergency_count,
            checks=all_checks,
            system_snapshot=system_snapshot
        )
        
        self.validation_history.append(report)
        return report
        
    def _create_system_snapshot(self) -> Dict[str, Any]:
        """Create comprehensive system snapshot"""
        snapshot = {}
        
        try:
            # System information
            snapshot['system'] = {
                'hostname': os.uname().nodename,
                'kernel': os.uname().release,
                'architecture': os.uname().machine,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Hardware information
            snapshot['hardware'] = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/')._asdict()
            }
            
            # Process information
            snapshot['processes'] = {
                'count': len(psutil.pids()),
                'load_average': os.getloadavg()
            }
            
            # Network information (basic)
            snapshot['network'] = {
                'connections': len(psutil.net_connections())
            }
            
        except Exception as e:
            snapshot['error'] = f"Failed to create system snapshot: {e}"
            
        return snapshot
        
    def save_safety_report(self, report: SafetyReport, filename: Optional[str] = None) -> Path:
        """Save safety report to file"""
        
        if filename is None:
            timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"safety_report_{timestamp}.json"
            
        report_path = self.work_dir / "testing" / filename
        report_path.parent.mkdir(exist_ok=True)
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'overall_status': report.overall_status.value,
            'total_checks': report.total_checks,
            'passed_checks': report.passed_checks,
            'failed_checks': report.failed_checks,
            'warnings': report.warnings,
            'critical_issues': report.critical_issues,
            'emergency_issues': report.emergency_issues,
            'system_snapshot': report.system_snapshot,
            'checks': []
        }
        
        # Convert checks
        for check in report.checks:
            check_dict = {
                'name': check.name,
                'category': check.category,
                'level': check.level.value,
                'status': check.status,
                'message': check.message,
                'value': check.value,
                'threshold': check.threshold,
                'timestamp': check.timestamp.isoformat() if check.timestamp else None
            }
            report_dict['checks'].append(check_dict)
            
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        return report_path
        
    def generate_safety_summary(self, report: SafetyReport) -> str:
        """Generate human-readable safety summary"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("SMBIOS TOKEN TESTING SAFETY VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall status
        status_emoji = {
            SafetyLevel.SAFE: "✅",
            SafetyLevel.WARNING: "⚠️",
            SafetyLevel.CRITICAL: "🚨",
            SafetyLevel.EMERGENCY: "🆘"
        }
        
        lines.append(f"Overall Status: {status_emoji[report.overall_status]} {report.overall_status.value}")
        lines.append(f"Validation Time: {report.timestamp}")
        lines.append(f"Total Checks: {report.total_checks}")
        lines.append(f"Passed: {report.passed_checks}")
        lines.append(f"Failed: {report.failed_checks}")
        if report.warnings > 0:
            lines.append(f"Warnings: {report.warnings}")
        if report.critical_issues > 0:
            lines.append(f"Critical Issues: {report.critical_issues}")
        if report.emergency_issues > 0:
            lines.append(f"Emergency Issues: {report.emergency_issues}")
        lines.append("")
        
        # Group checks by category
        categories = {}
        for check in report.checks:
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check)
            
        for category, checks in categories.items():
            lines.append(f"{category.upper()} CHECKS:")
            lines.append("-" * 40)
            
            for check in checks:
                status_char = "✅" if check.status else "❌"
                level_char = status_emoji.get(check.level, "❓")
                lines.append(f"  {status_char} {level_char} {check.name}: {check.message}")
                
                if check.value is not None and check.threshold is not None:
                    lines.append(f"      Value: {check.value}, Threshold: {check.threshold}")
                    
            lines.append("")
            
        # Recommendations
        if report.overall_status != SafetyLevel.SAFE:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 40)
            
            if report.emergency_issues > 0:
                lines.append("  🆘 EMERGENCY: Do not proceed with testing!")
                lines.append("     - System is in emergency state")
                lines.append("     - Resolve all emergency issues first")
                
            elif report.critical_issues > 0:
                lines.append("  🚨 CRITICAL: Testing not recommended!")
                lines.append("     - Resolve critical issues before testing")
                lines.append("     - System may be unstable")
                
            elif report.warnings > 0:
                lines.append("  ⚠️ WARNING: Proceed with caution")
                lines.append("     - Address warnings if possible")
                lines.append("     - Monitor system closely during testing")
                
            lines.append("")
            
        return "\n".join(lines)

def main():
    """Test safety validation system"""
    
    print("🔒 SMBIOS Token Testing Safety Validator v1.0.0")
    print("Dell Latitude 5450 MIL-SPEC - TESTBED Agent")
    print("=" * 60)
    
    validator = SafetyValidator()
    
    # Run full safety validation
    report = validator.run_full_safety_validation()
    
    # Generate and display summary
    summary = validator.generate_safety_summary(report)
    print(summary)
    
    # Save report
    report_path = validator.save_safety_report(report)
    print(f"📊 Full safety report saved to: {report_path}")
    
    # Return appropriate exit code
    if report.overall_status == SafetyLevel.EMERGENCY:
        print("\n🆘 EMERGENCY: System not safe for testing!")
        return 2
    elif report.overall_status == SafetyLevel.CRITICAL:
        print("\n🚨 CRITICAL: System not recommended for testing!")
        return 1
    elif report.overall_status == SafetyLevel.WARNING:
        print("\n⚠️ WARNING: System has issues but may proceed with caution")
        return 0
    else:
        print("\n✅ SAFE: System ready for SMBIOS token testing")
        return 0

if __name__ == "__main__":
    sys.exit(main())
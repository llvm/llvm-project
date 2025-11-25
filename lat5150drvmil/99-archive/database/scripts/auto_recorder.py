#!/usr/bin/env python3
"""
DSMIL Token Testing Auto-Recording System
Automatic capture of test operations, system metrics, and kernel messages
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import time
import threading
import subprocess
import re
import json
import uuid
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Add database backend to path
sys.path.insert(0, '/home/john/LAT5150DRVMIL/database/backends')
from database_backend import DatabaseBackend, TokenTestResult, ThermalReading, SystemMetric, DSMILResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/home/john/LAT5150DRVMIL/database/scripts/auto_recorder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KernelMessage:
    """Kernel message structure"""
    timestamp: float
    level: str
    subsystem: str
    message: str
    raw_message: str

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, session_id: str, db_backend: DatabaseBackend):
        self.session_id = session_id
        self.db = db_backend
        self.running = False
        self.monitor_thread = None
        self.interval = 1.0  # 1 second intervals
        
    def start(self):
        """Start system monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop(self):
        """Stop system monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_thermal_readings()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(self.interval)
                
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = os.getloadavg()
            
            # System uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
                uptime_hours = uptime_seconds / 3600
                
            # Process count
            process_count = len(psutil.pids())
            
            metric = SystemMetric(
                metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                test_id=None,  # Will be associated later if needed
                session_id=self.session_id,
                metric_timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                system_load_1min=load_avg[0],
                system_load_5min=load_avg[1],
                system_load_15min=load_avg[2],
                uptime_hours=uptime_hours,
                process_count=process_count
            )
            
            self.db.record_system_metric(metric)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            
    def _collect_thermal_readings(self):
        """Collect thermal sensor readings"""
        try:
            # Try multiple sources for thermal data
            thermal_data = {}
            
            # hwmon sensors
            hwmon_base = Path("/sys/class/hwmon")
            if hwmon_base.exists():
                for hwmon_dir in hwmon_base.iterdir():
                    if hwmon_dir.is_dir():
                        name_file = hwmon_dir / "name"
                        if name_file.exists():
                            sensor_name = name_file.read_text().strip()
                            
                            # Look for temperature inputs
                            for temp_file in hwmon_dir.glob("temp*_input"):
                                try:
                                    temp_millicelsius = int(temp_file.read_text().strip())
                                    temp_celsius = temp_millicelsius / 1000.0
                                    
                                    # Get critical and warning temperatures if available
                                    crit_file = temp_file.parent / temp_file.name.replace("_input", "_crit")
                                    warn_file = temp_file.parent / temp_file.name.replace("_input", "_max")
                                    
                                    crit_temp = None
                                    warn_temp = None
                                    
                                    if crit_file.exists():
                                        try:
                                            crit_temp = int(crit_file.read_text().strip()) / 1000.0
                                        except:
                                            pass
                                            
                                    if warn_file.exists():
                                        try:
                                            warn_temp = int(warn_file.read_text().strip()) / 1000.0
                                        except:
                                            pass
                                            
                                    # Determine thermal state
                                    thermal_state = "normal"
                                    if crit_temp and temp_celsius > crit_temp:
                                        thermal_state = "critical"
                                    elif warn_temp and temp_celsius > warn_temp:
                                        thermal_state = "warning"
                                    elif temp_celsius > 95:  # MIL-SPEC warning threshold
                                        thermal_state = "warning"
                                    elif temp_celsius > 100:  # MIL-SPEC critical threshold
                                        thermal_state = "critical"
                                        
                                    sensor_full_name = f"{sensor_name}_{temp_file.stem}"
                                    thermal_data[sensor_full_name] = {
                                        'temperature': temp_celsius,
                                        'critical_temp': crit_temp,
                                        'warning_temp': warn_temp,
                                        'thermal_state': thermal_state
                                    }
                                    
                                except Exception as e:
                                    logger.debug(f"Could not read {temp_file}: {str(e)}")
                                    
            # Also try ACPI thermal zone
            acpi_thermal_base = Path("/sys/class/thermal")
            if acpi_thermal_base.exists():
                for thermal_zone in acpi_thermal_base.glob("thermal_zone*"):
                    try:
                        type_file = thermal_zone / "type"
                        temp_file = thermal_zone / "temp"
                        
                        if type_file.exists() and temp_file.exists():
                            zone_type = type_file.read_text().strip()
                            temp_millicelsius = int(temp_file.read_text().strip())
                            temp_celsius = temp_millicelsius / 1000.0
                            
                            thermal_state = "normal"
                            if temp_celsius > 95:
                                thermal_state = "warning"
                            elif temp_celsius > 100:
                                thermal_state = "critical"
                                
                            thermal_data[f"acpi_{zone_type}"] = {
                                'temperature': temp_celsius,
                                'thermal_state': thermal_state
                            }
                            
                    except Exception as e:
                        logger.debug(f"Could not read ACPI thermal zone {thermal_zone}: {str(e)}")
                        
            # Record all thermal readings
            timestamp = time.time()
            for sensor_name, data in thermal_data.items():
                reading = ThermalReading(
                    reading_id=f"thermal_{uuid.uuid4().hex[:8]}",
                    test_id=None,
                    session_id=self.session_id,
                    reading_timestamp=timestamp,
                    sensor_name=sensor_name,
                    temperature_celsius=data['temperature'],
                    critical_temp=data.get('critical_temp'),
                    warning_temp=data.get('warning_temp'),
                    thermal_state=data['thermal_state'],
                    thermal_throttling=data['thermal_state'] in ['warning', 'critical']
                )
                
                self.db.record_thermal_reading(reading)
                
        except Exception as e:
            logger.error(f"Failed to collect thermal readings: {str(e)}")

class KernelMessageMonitor:
    """Monitor kernel messages for DSMIL and related events"""
    
    def __init__(self, session_id: str, db_backend: DatabaseBackend):
        self.session_id = session_id
        self.db = db_backend
        self.running = False
        self.monitor_thread = None
        self.dmesg_process = None
        
        # Message parsing patterns
        self.patterns = {
            'dsmil': re.compile(r'dsmil|DSMIL|72dev', re.IGNORECASE),
            'smbios': re.compile(r'smbios|SMBIOS|token', re.IGNORECASE),
            'thermal': re.compile(r'thermal|temperature|overheat|throttle', re.IGNORECASE),
            'acpi': re.compile(r'acpi|ACPI|dsdt|ssdt', re.IGNORECASE),
            'dell': re.compile(r'dell|Dell|DELL|latitude', re.IGNORECASE)
        }
        
    def start(self):
        """Start kernel message monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_kernel_messages)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Kernel message monitoring started")
        
    def stop(self):
        """Stop kernel message monitoring"""
        self.running = False
        if self.dmesg_process:
            self.dmesg_process.terminate()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Kernel message monitoring stopped")
        
    def _monitor_kernel_messages(self):
        """Monitor kernel messages using dmesg -w"""
        try:
            # Start dmesg follow process
            self.dmesg_process = subprocess.Popen(
                ['dmesg', '-w', '--time-format', 'iso'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            while self.running and self.dmesg_process.poll() is None:
                line = self.dmesg_process.stdout.readline()
                if line:
                    self._process_kernel_message(line.strip())
                    
        except Exception as e:
            logger.error(f"Kernel message monitoring error: {str(e)}")
            
    def _process_kernel_message(self, message: str):
        """Process a kernel message line"""
        try:
            # Parse timestamp and message
            if message.startswith('['):
                # Format: [timestamp] message
                parts = message.split('] ', 1)
                if len(parts) == 2:
                    timestamp_str = parts[0][1:]  # Remove '['
                    msg_content = parts[1]
                    
                    # Parse timestamp
                    try:
                        msg_timestamp = datetime.fromisoformat(timestamp_str.replace('T', ' ')).timestamp()
                    except:
                        msg_timestamp = time.time()
                        
                    # Determine log level and subsystem
                    log_level = self._extract_log_level(msg_content)
                    subsystem = self._extract_subsystem(msg_content)
                    
                    # Check if message is relevant
                    if self._is_relevant_message(msg_content):
                        # Store in database
                        message_id = f"kmsg_{uuid.uuid4().hex[:8]}"
                        
                        with self.db._get_sqlite_connection() as conn:
                            conn.execute("""
                                INSERT INTO kernel_messages (
                                    message_id, test_id, session_id, message_timestamp,
                                    log_level, subsystem, message_text, raw_message
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                message_id, None, self.session_id, msg_timestamp,
                                log_level, subsystem, msg_content, message
                            ))
                            conn.commit()
                            
                        logger.debug(f"Recorded kernel message: {subsystem} - {msg_content[:100]}")
                        
        except Exception as e:
            logger.error(f"Error processing kernel message: {str(e)}")
            
    def _extract_log_level(self, message: str) -> str:
        """Extract log level from kernel message"""
        # Common kernel log level indicators
        if 'emerg' in message.lower() or 'panic' in message.lower():
            return 'emerg'
        elif 'alert' in message.lower():
            return 'alert'
        elif 'crit' in message.lower() or 'critical' in message.lower():
            return 'crit'
        elif 'err' in message.lower() or 'error' in message.lower():
            return 'err'
        elif 'warn' in message.lower() or 'warning' in message.lower():
            return 'warn'
        elif 'notice' in message.lower():
            return 'notice'
        elif 'info' in message.lower():
            return 'info'
        else:
            return 'debug'
            
    def _extract_subsystem(self, message: str) -> str:
        """Extract subsystem from kernel message"""
        for subsystem, pattern in self.patterns.items():
            if pattern.search(message):
                return subsystem
        return 'kernel'
        
    def _is_relevant_message(self, message: str) -> bool:
        """Check if message is relevant for DSMIL testing"""
        # Always capture messages matching our patterns
        for pattern in self.patterns.values():
            if pattern.search(message):
                return True
                
        # Also capture error/warning messages
        msg_lower = message.lower()
        if any(word in msg_lower for word in ['error', 'warning', 'fail', 'timeout', 'invalid']):
            return True
            
        return False

class DSMILResponseMonitor:
    """Monitor DSMIL kernel module responses and device state changes"""
    
    def __init__(self, session_id: str, db_backend: DatabaseBackend):
        self.session_id = session_id
        self.db = db_backend
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start DSMIL response monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_dsmil_responses)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("DSMIL response monitoring started")
        
    def stop(self):
        """Stop DSMIL response monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("DSMIL response monitoring stopped")
        
    def _monitor_dsmil_responses(self):
        """Monitor for DSMIL device responses"""
        while self.running:
            try:
                # Check DSMIL module status
                self._check_module_status()
                
                # Check for device state changes
                self._check_device_states()
                
                # Check memory mappings
                self._check_memory_mappings()
                
                time.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"DSMIL response monitoring error: {str(e)}")
                time.sleep(5.0)
                
    def _check_module_status(self):
        """Check DSMIL kernel module status"""
        try:
            # Check if module is loaded
            with open('/proc/modules', 'r') as f:
                modules = f.read()
                
            dsmil_loaded = 'dsmil_72dev' in modules
            
            # Check module parameters
            param_dir = Path('/sys/module/dsmil_72dev/parameters')
            if param_dir.exists():
                for param_file in param_dir.iterdir():
                    if param_file.is_file():
                        param_name = param_file.name
                        param_value = param_file.read_text().strip()
                        logger.debug(f"DSMIL parameter {param_name}: {param_value}")
                        
        except Exception as e:
            logger.debug(f"Could not check module status: {str(e)}")
            
    def _check_device_states(self):
        """Check for DSMIL device state changes"""
        try:
            # Check sysfs entries
            dsmil_sysfs = Path('/sys/devices/platform/dsmil-72dev')
            if dsmil_sysfs.exists():
                # Look for group and device status files
                for group_dir in dsmil_sysfs.glob('group*'):
                    group_id = int(group_dir.name.replace('group', ''))
                    
                    status_file = group_dir / 'status'
                    if status_file.exists():
                        try:
                            status = status_file.read_text().strip()
                            
                            # Record state change if different from last known
                            response = DSMILResponse(
                                response_id=f"dsmil_{uuid.uuid4().hex[:8]}",
                                test_id=None,
                                session_id=self.session_id,
                                response_timestamp=time.time(),
                                group_id=group_id,
                                device_id=None,
                                response_type="group_status",
                                new_state=status,
                                correlation_strength=0.8
                            )
                            
                            self.db.record_dsmil_response(response)
                            
                        except Exception as e:
                            logger.debug(f"Could not read group {group_id} status: {str(e)}")
                            
        except Exception as e:
            logger.debug(f"Could not check device states: {str(e)}")
            
    def _check_memory_mappings(self):
        """Check for memory mapping changes"""
        try:
            # Check /proc/iomem for DSMIL-related mappings
            with open('/proc/iomem', 'r') as f:
                iomem_lines = f.readlines()
                
            for line in iomem_lines:
                if 'dsmil' in line.lower() or '52000000' in line:  # DSMIL base address
                    # Parse memory range
                    parts = line.strip().split(':', 1)
                    if len(parts) == 2:
                        addr_range = parts[0].strip()
                        description = parts[1].strip()
                        
                        logger.debug(f"DSMIL memory mapping: {addr_range} - {description}")
                        
        except Exception as e:
            logger.debug(f"Could not check memory mappings: {str(e)}")

class AutoRecorder:
    """Main auto-recording system coordinator"""
    
    def __init__(self, base_path: str = "/home/john/LAT5150DRVMIL/database"):
        self.base_path = Path(base_path)
        self.db = DatabaseBackend(str(base_path))
        
        # Monitoring components
        self.system_monitor = None
        self.kernel_monitor = None
        self.dsmil_monitor = None
        
        # State tracking
        self.current_session = None
        self.running = False
        
        # Token test tracking
        self.active_token_tests = {}  # test_id -> TokenTestResult
        self.token_test_callbacks = []
        
    def start_session(self, session_name: str, session_type: str, 
                     operator: Optional[str] = None) -> str:
        """Start a new recording session"""
        if self.current_session:
            logger.warning(f"Session {self.current_session} already active, closing it")
            self.stop_session()
            
        self.current_session = self.db.create_session(session_name, session_type, operator)
        
        # Start all monitors
        self.system_monitor = SystemMonitor(self.current_session, self.db)
        self.kernel_monitor = KernelMessageMonitor(self.current_session, self.db)
        self.dsmil_monitor = DSMILResponseMonitor(self.current_session, self.db)
        
        self.system_monitor.start()
        self.kernel_monitor.start()
        self.dsmil_monitor.start()
        
        self.running = True
        
        logger.info(f"Started recording session: {self.current_session}")
        return self.current_session
        
    def stop_session(self, status: str = "completed", notes: Optional[str] = None):
        """Stop the current recording session"""
        if not self.current_session:
            logger.warning("No active session to stop")
            return
            
        # Stop all monitors
        if self.system_monitor:
            self.system_monitor.stop()
        if self.kernel_monitor:
            self.kernel_monitor.stop()
        if self.dsmil_monitor:
            self.dsmil_monitor.stop()
            
        # Close session in database
        self.db.close_session(self.current_session, status, notes)
        
        logger.info(f"Stopped recording session: {self.current_session}")
        self.current_session = None
        self.running = False
        
    def record_token_operation(self, token_id: int, hex_id: str, 
                              access_method: str, operation_type: str,
                              set_value: Optional[str] = None) -> str:
        """Record the start of a token operation"""
        if not self.current_session:
            raise RuntimeError("No active recording session")
            
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()
        
        # Get initial value if possible
        initial_value = self._get_token_value(token_id, hex_id, access_method)
        
        # Create test result record
        result = TokenTestResult(
            test_id=test_id,
            session_id=self.current_session,
            token_id=token_id,
            hex_id=hex_id,
            group_id=token_id // 12 if token_id >= 1152 else 0,  # Calculate group from token_id
            device_id=(token_id - 1152) % 12 if token_id >= 1152 else 0,  # Calculate device
            test_timestamp=timestamp,
            access_method=access_method,
            operation_type=operation_type,
            initial_value=initial_value,
            set_value=set_value
        )
        
        # Store for later completion
        self.active_token_tests[test_id] = result
        
        logger.info(f"Started token operation: {test_id} - {hex_id} ({operation_type})")
        return test_id
        
    def complete_token_operation(self, test_id: str, success: bool, 
                                final_value: Optional[str] = None,
                                error_code: Optional[str] = None,
                                error_message: Optional[str] = None,
                                notes: Optional[str] = None):
        """Complete a token operation record"""
        if test_id not in self.active_token_tests:
            logger.error(f"Unknown test_id: {test_id}")
            return
            
        result = self.active_token_tests[test_id]
        
        # Update result with completion data
        result.final_value = final_value
        result.success = success
        result.error_code = error_code
        result.error_message = error_message
        result.notes = notes
        result.test_duration_ms = int((time.time() - result.test_timestamp) * 1000)
        
        # Record in database
        self.db.record_token_test(result)
        
        # Remove from active tests
        del self.active_token_tests[test_id]
        
        logger.info(f"Completed token operation: {test_id} - {'SUCCESS' if success else 'FAILED'}")
        
    def _get_token_value(self, token_id: int, hex_id: str, access_method: str) -> Optional[str]:
        """Attempt to get current token value"""
        try:
            if access_method == "smbios-token-ctl":
                result = subprocess.run(
                    ['smbios-token-ctl', '--get-token', str(token_id)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get token value for {hex_id}: {str(e)}")
            
        return None
        
    def create_backup(self) -> str:
        """Create a backup of all recorded data"""
        return self.db.create_backup()
        
    def get_session_summary(self) -> Optional[Dict]:
        """Get summary of current session"""
        if not self.current_session:
            return None
        return self.db.get_session_summary(self.current_session)
        
    def register_token_test_callback(self, callback: Callable[[TokenTestResult], None]):
        """Register a callback for token test events"""
        self.token_test_callbacks.append(callback)

# Context manager for easy session management
class RecordingSession:
    """Context manager for auto-recording sessions"""
    
    def __init__(self, session_name: str, session_type: str, operator: Optional[str] = None):
        self.session_name = session_name
        self.session_type = session_type
        self.operator = operator
        self.recorder = None
        self.session_id = None
        
    def __enter__(self):
        self.recorder = AutoRecorder()
        self.session_id = self.recorder.start_session(
            self.session_name, self.session_type, self.operator
        )
        return self.recorder
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder:
            status = "failed" if exc_type else "completed"
            notes = str(exc_val) if exc_val else None
            self.recorder.stop_session(status, notes)

if __name__ == "__main__":
    # Example usage
    with RecordingSession("Test Recording", "single", "auto_test") as recorder:
        # Simulate token operation
        test_id = recorder.record_token_operation(
            token_id=1152,
            hex_id="0x480",
            access_method="smbios-token-ctl",
            operation_type="read"
        )
        
        # Simulate some work
        time.sleep(5)
        
        # Complete the operation
        recorder.complete_token_operation(
            test_id=test_id,
            success=True,
            final_value="1",
            notes="Test operation completed successfully"
        )
        
        print(f"Recording session completed: {recorder.current_session}")
        
        # Get session summary
        summary = recorder.get_session_summary()
        print(f"Session summary: {summary}")
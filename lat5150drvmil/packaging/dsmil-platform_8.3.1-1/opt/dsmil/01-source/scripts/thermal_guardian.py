#!/usr/bin/env python3
"""
Thermal Guardian - Advanced Thermal Management System
Agent 3 Implementation for Dell LAT5150DRVMIL

A production-ready thermal management system that prevents thermal shutdown
while maintaining maximum performance through predictive modeling and
graduated response control.

Author: Thermal Guardian Agent Team - Agent 3
Version: 1.0
"""

import os
import sys
import time
import json
import logging
import argparse
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
from collections import deque, defaultdict

# Hardware interface constants
SYSFS_THERMAL = Path("/sys/class/thermal")
SYSFS_HWMON = Path("/sys/class/hwmon")
INTEL_PSTATE = Path("/sys/devices/system/cpu/intel_pstate")
CPUFREQ_BASE = Path("/sys/devices/system/cpu")

@dataclass
class ThermalConfig:
    """Configuration parameters for thermal management"""
    # Temperature thresholds (Celsius)
    temp_normal: float = 85.0
    temp_warm: float = 90.0
    temp_hot: float = 95.0
    temp_critical: float = 100.0
    temp_emergency: float = 103.0
    temp_shutdown: float = 105.0
    
    # Hysteresis settings
    hysteresis_normal: float = 3.0
    hysteresis_warm: float = 2.0
    hysteresis_hot: float = 2.0
    hysteresis_critical: float = 1.0
    
    # Control parameters
    fan_min_pwm: int = 30
    fan_max_pwm: int = 255
    cpu_min_freq_pct: int = 20
    cpu_max_freq_pct: int = 100
    
    # Monitoring settings
    poll_interval: float = 1.0
    prediction_window: int = 10
    sensor_timeout: float = 5.0
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "/var/log/thermal_guardian.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

class ThermalSensor:
    """Individual thermal sensor management"""
    
    def __init__(self, name: str, path: str, scale: float = 1000.0):
        self.name = name
        self.path = Path(path)
        self.scale = scale
        self.last_temp = None
        self.last_read_time = None
        self.read_errors = 0
        self.max_errors = 5
        
    def read_temperature(self) -> Optional[float]:
        """Read temperature from sensor with error handling"""
        try:
            if not self.path.exists():
                return None
                
            with open(self.path, 'r') as f:
                raw_value = int(f.read().strip())
                temp = raw_value / self.scale
                
            self.last_temp = temp
            self.last_read_time = time.time()
            self.read_errors = 0
            return temp
            
        except (FileNotFoundError, ValueError, PermissionError) as e:
            self.read_errors += 1
            logging.warning(f"Error reading sensor {self.name}: {e}")
            return None
            
    def is_healthy(self) -> bool:
        """Check if sensor is functioning properly"""
        return (self.read_errors < self.max_errors and 
                self.last_read_time is not None and
                time.time() - self.last_read_time < 30)

class FanController:
    """Dell SMM fan control interface"""
    
    def __init__(self):
        self.fan_paths = {}
        self.current_pwm = {}
        self.discover_fans()
        
    def discover_fans(self):
        """Discover available Dell SMM fan interfaces"""
        for hwmon_path in SYSFS_HWMON.glob("hwmon*"):
            name_file = hwmon_path / "name"
            if name_file.exists():
                try:
                    with open(name_file, 'r') as f:
                        name = f.read().strip()
                    
                    if name in ["dell_smm", "i8k"]:
                        # Find PWM controls
                        for pwm_file in hwmon_path.glob("pwm*"):
                            fan_id = pwm_file.name[3:]  # Remove 'pwm' prefix
                            self.fan_paths[fan_id] = pwm_file
                            logging.info(f"Found fan control: {pwm_file}")
                            
                except Exception as e:
                    logging.debug(f"Error checking hwmon {hwmon_path}: {e}")
                    
    def set_fan_speed(self, fan_id: str, pwm_value: int) -> bool:
        """Set fan PWM value (0-255)"""
        if fan_id not in self.fan_paths:
            return False
            
        try:
            pwm_value = max(0, min(255, pwm_value))
            with open(self.fan_paths[fan_id], 'w') as f:
                f.write(str(pwm_value))
            self.current_pwm[fan_id] = pwm_value
            return True
            
        except (PermissionError, FileNotFoundError) as e:
            logging.error(f"Failed to set fan {fan_id} PWM: {e}")
            return False
            
    def get_fan_speed(self, fan_id: str) -> Optional[int]:
        """Get current fan PWM value"""
        if fan_id not in self.fan_paths:
            return None
            
        try:
            with open(self.fan_paths[fan_id], 'r') as f:
                return int(f.read().strip())
        except Exception:
            return None

class CPUFrequencyController:
    """Intel P-State CPU frequency control"""
    
    def __init__(self):
        self.max_freq_path = INTEL_PSTATE / "max_perf_pct"
        self.min_freq_path = INTEL_PSTATE / "min_perf_pct"
        self.no_turbo_path = INTEL_PSTATE / "no_turbo"
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Intel P-State is available"""
        return (INTEL_PSTATE.exists() and 
                self.max_freq_path.exists() and
                self.min_freq_path.exists())
                
    def set_max_frequency_pct(self, percentage: int) -> bool:
        """Set maximum CPU frequency as percentage of base frequency"""
        if not self.is_available:
            return False
            
        try:
            percentage = max(1, min(100, percentage))
            with open(self.max_freq_path, 'w') as f:
                f.write(str(percentage))
            return True
            
        except (PermissionError, FileNotFoundError) as e:
            logging.error(f"Failed to set max CPU frequency: {e}")
            return False
            
    def disable_turbo(self) -> bool:
        """Disable Intel Turbo Boost"""
        if not self.is_available or not self.no_turbo_path.exists():
            return False
            
        try:
            with open(self.no_turbo_path, 'w') as f:
                f.write("1")
            return True
            
        except (PermissionError, FileNotFoundError) as e:
            logging.error(f"Failed to disable turbo: {e}")
            return False
            
    def enable_turbo(self) -> bool:
        """Enable Intel Turbo Boost"""
        if not self.is_available or not self.no_turbo_path.exists():
            return False
            
        try:
            with open(self.no_turbo_path, 'w') as f:
                f.write("0")
            return True
            
        except (PermissionError, FileNotFoundError) as e:
            logging.error(f"Failed to enable turbo: {e}")
            return False

class ThermalPredictor:
    """Predictive thermal modeling system"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.temp_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        
    def add_sample(self, temperature: float):
        """Add new temperature sample"""
        current_time = time.time()
        self.temp_history.append(temperature)
        self.time_history.append(current_time)
        
    def predict_temperature(self, seconds_ahead: float = 5.0) -> Optional[float]:
        """Predict temperature N seconds in the future"""
        if len(self.temp_history) < 3:
            return None
            
        # Calculate temperature rate of change
        try:
            temp_changes = []
            time_deltas = []
            
            for i in range(1, len(self.temp_history)):
                temp_delta = self.temp_history[i] - self.temp_history[i-1]
                time_delta = self.time_history[i] - self.time_history[i-1]
                
                if time_delta > 0:
                    temp_changes.append(temp_delta / time_delta)
                    time_deltas.append(time_delta)
                    
            if not temp_changes:
                return self.temp_history[-1]
                
            # Weighted average favoring recent changes
            weights = [2**i for i in range(len(temp_changes))]
            weighted_rate = sum(rate * weight for rate, weight in zip(temp_changes, weights)) / sum(weights)
            
            # Apply exponential smoothing for stability
            current_temp = self.temp_history[-1]
            predicted_temp = current_temp + (weighted_rate * seconds_ahead)
            
            return predicted_temp
            
        except Exception as e:
            logging.debug(f"Prediction error: {e}")
            return self.temp_history[-1] if self.temp_history else None

class ThermalManager:
    """Main thermal management system"""
    
    def __init__(self, config: ThermalConfig):
        self.config = config
        self.sensors = {}
        self.fan_controller = FanController()
        self.cpu_controller = CPUFrequencyController()
        self.predictor = ThermalPredictor(config.prediction_window)
        
        # State tracking
        self.current_state = "normal"
        self.state_history = deque(maxlen=100)
        self.emergency_triggered = False
        self.shutdown_pending = False
        
        # Performance tracking
        self.throttle_start_time = None
        self.total_throttle_time = 0
        
        # Initialize sensors
        self._discover_sensors()
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _discover_sensors(self):
        """Discover and initialize thermal sensors"""
        sensor_configs = [
            # Primary sensors for Dell LAT5150DRVMIL
            ("x86_pkg_temp", "/sys/class/thermal/thermal_zone*/temp", 1000),
            ("coretemp_package", "/sys/class/hwmon/hwmon*/temp1_input", 1000),
            ("dell_tcpu", "/sys/class/hwmon/hwmon*/temp*_input", 1000),
            ("acpi_thermal", "/sys/class/thermal/thermal_zone*/temp", 1000),
        ]
        
        for sensor_name, pattern, scale in sensor_configs:
            paths = list(Path("/").glob(pattern.lstrip("/")))
            for path in paths:
                try:
                    # Verify sensor exists and is readable
                    if path.exists():
                        test_sensor = ThermalSensor(f"{sensor_name}_{path.parent.name}", str(path), scale)
                        test_temp = test_sensor.read_temperature()
                        
                        if test_temp is not None and 20 <= test_temp <= 120:  # Reasonable range
                            self.sensors[test_sensor.name] = test_sensor
                            logging.info(f"Initialized sensor: {test_sensor.name}")
                            
                except Exception as e:
                    logging.debug(f"Failed to initialize sensor at {path}: {e}")
                    
        if not self.sensors:
            logging.error("No thermal sensors found! System may not be protected.")
            
    def _setup_logging(self):
        """Configure logging system"""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.config.log_file,
            maxBytes=self.config.log_max_size,
            backupCount=self.config.log_backup_count
        )
        file_handler.setFormatter(log_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        
        # Configure root logger
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().addHandler(console_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_pending = True
        
    def read_all_sensors(self) -> Dict[str, float]:
        """Read all available sensors"""
        temperatures = {}
        
        for name, sensor in self.sensors.items():
            temp = sensor.read_temperature()
            if temp is not None:
                temperatures[name] = temp
                
        return temperatures
        
    def get_max_temperature(self) -> Tuple[float, str]:
        """Get the maximum temperature from all sensors"""
        temps = self.read_all_sensors()
        
        if not temps:
            logging.error("No temperature readings available!")
            return 0.0, "unknown"
            
        max_temp = max(temps.values())
        max_sensor = max(temps, key=temps.get)
        
        return max_temp, max_sensor
        
    def determine_thermal_state(self, temperature: float, predicted_temp: Optional[float] = None) -> str:
        """Determine thermal state with hysteresis"""
        current_state = self.current_state
        temp_to_check = max(temperature, predicted_temp or 0)
        
        # Emergency shutdown check
        if temp_to_check >= self.config.temp_shutdown:
            return "shutdown"
            
        # Use hysteresis to prevent oscillation
        if current_state == "normal":
            if temp_to_check >= self.config.temp_warm:
                return "warm"
        elif current_state == "warm":
            if temp_to_check >= self.config.temp_hot:
                return "hot"
            elif temp_to_check < self.config.temp_normal:
                return "normal"
        elif current_state == "hot":
            if temp_to_check >= self.config.temp_critical:
                return "critical"
            elif temp_to_check < (self.config.temp_warm - self.config.hysteresis_warm):
                return "warm"
        elif current_state == "critical":
            if temp_to_check >= self.config.temp_emergency:
                return "emergency"
            elif temp_to_check < (self.config.temp_hot - self.config.hysteresis_hot):
                return "hot"
        elif current_state == "emergency":
            if temp_to_check < (self.config.temp_critical - self.config.hysteresis_critical):
                return "critical"
                
        return current_state
        
    def apply_thermal_response(self, state: str, temperature: float):
        """Apply graduated thermal response"""
        
        if state == "normal":
            # Normal operation - restore full performance
            if self.throttle_start_time:
                self.total_throttle_time += time.time() - self.throttle_start_time
                self.throttle_start_time = None
                
            self.cpu_controller.set_max_frequency_pct(self.config.cpu_max_freq_pct)
            self.cpu_controller.enable_turbo()
            
            # Set fans to minimum
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, self.config.fan_min_pwm)
                
        elif state == "warm":
            # Increase fan speed
            fan_pwm = int(self.config.fan_min_pwm + 
                         (self.config.fan_max_pwm - self.config.fan_min_pwm) * 0.3)
            
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, fan_pwm)
                
        elif state == "hot":
            # Increase fans more, slight CPU throttle
            if not self.throttle_start_time:
                self.throttle_start_time = time.time()
                
            fan_pwm = int(self.config.fan_min_pwm + 
                         (self.config.fan_max_pwm - self.config.fan_min_pwm) * 0.6)
            
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, fan_pwm)
                
            self.cpu_controller.set_max_frequency_pct(80)
            
        elif state == "critical":
            # Maximum fans, significant CPU throttle
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, self.config.fan_max_pwm)
                
            self.cpu_controller.set_max_frequency_pct(60)
            self.cpu_controller.disable_turbo()
            
        elif state == "emergency":
            # Emergency measures
            self.emergency_triggered = True
            
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, self.config.fan_max_pwm)
                
            self.cpu_controller.set_max_frequency_pct(self.config.cpu_min_freq_pct)
            self.cpu_controller.disable_turbo()
            
            logging.critical(f"EMERGENCY THERMAL STATE: {temperature:.1f}°C")
            
        elif state == "shutdown":
            # Initiate emergency shutdown
            logging.critical(f"THERMAL SHUTDOWN INITIATED: {temperature:.1f}°C")
            self.emergency_shutdown()
            
    def emergency_shutdown(self):
        """Initiate emergency system shutdown"""
        self.shutdown_pending = True
        
        # Log the shutdown
        logging.critical("EMERGENCY THERMAL SHUTDOWN - TEMPERATURE TOO HIGH")
        
        # Attempt to shut down system
        try:
            subprocess.run(["/sbin/shutdown", "-h", "now", "Thermal emergency"], 
                         timeout=10)
        except Exception as e:
            logging.error(f"Failed to initiate shutdown: {e}")
            # Force kill if shutdown fails
            os.system("poweroff")
            
    def run_monitoring_cycle(self):
        """Single monitoring cycle"""
        # Read sensors
        max_temp, max_sensor = self.get_max_temperature()
        
        if max_temp == 0:
            logging.warning("No valid temperature readings")
            return
            
        # Update predictor
        self.predictor.add_sample(max_temp)
        predicted_temp = self.predictor.predict_temperature(5.0)
        
        # Determine thermal state
        new_state = self.determine_thermal_state(max_temp, predicted_temp)
        
        # State change logging
        if new_state != self.current_state:
            logging.info(f"Thermal state change: {self.current_state} -> {new_state} "
                        f"(Current: {max_temp:.1f}°C, Predicted: {predicted_temp:.1f}°C)")
            
            self.state_history.append({
                'timestamp': datetime.now().isoformat(),
                'from_state': self.current_state,
                'to_state': new_state,
                'temperature': max_temp,
                'predicted': predicted_temp,
                'sensor': max_sensor
            })
            
        self.current_state = new_state
        
        # Apply thermal response
        self.apply_thermal_response(new_state, max_temp)
        
        # Periodic detailed logging
        if int(time.time()) % 60 == 0:  # Every minute
            all_temps = self.read_all_sensors()
            temp_str = ", ".join([f"{name}: {temp:.1f}°C" for name, temp in all_temps.items()])
            logging.info(f"Thermal status: {new_state} - {temp_str}")
            
    def run(self):
        """Main monitoring loop"""
        logging.info("Thermal Guardian starting...")
        logging.info(f"Monitoring {len(self.sensors)} sensors")
        logging.info(f"Found {len(self.fan_controller.fan_paths)} fan controllers")
        logging.info(f"CPU frequency control: {'Available' if self.cpu_controller.is_available else 'Not Available'}")
        
        try:
            while not self.shutdown_pending:
                loop_start = time.time()
                
                try:
                    self.run_monitoring_cycle()
                except Exception as e:
                    logging.error(f"Error in monitoring cycle: {e}")
                    
                # Maintain precise timing
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.config.poll_interval - loop_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user")
        except Exception as e:
            logging.critical(f"Fatal error in thermal guardian: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup and restore system state"""
        logging.info("Thermal Guardian shutting down...")
        
        try:
            # Restore normal operation
            self.cpu_controller.set_max_frequency_pct(self.config.cpu_max_freq_pct)
            self.cpu_controller.enable_turbo()
            
            # Set fans to automatic control
            for fan_id in self.fan_controller.fan_paths:
                self.fan_controller.set_fan_speed(fan_id, self.config.fan_min_pwm)
                
            # Log performance impact
            if self.throttle_start_time:
                self.total_throttle_time += time.time() - self.throttle_start_time
                
            if self.total_throttle_time > 0:
                logging.info(f"Total throttle time this session: {self.total_throttle_time:.1f} seconds")
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            
        logging.info("Thermal Guardian stopped")

    def get_status(self) -> Dict:
        """Get current system status"""
        temps = self.read_all_sensors()
        max_temp, max_sensor = self.get_max_temperature()
        predicted = self.predictor.predict_temperature()
        
        return {
            'current_state': self.current_state,
            'max_temperature': max_temp,
            'max_sensor': max_sensor,
            'predicted_temperature': predicted,
            'all_temperatures': temps,
            'emergency_triggered': self.emergency_triggered,
            'total_throttle_time': self.total_throttle_time,
            'sensors_healthy': sum(1 for s in self.sensors.values() if s.is_healthy()),
            'total_sensors': len(self.sensors),
            'fan_controllers': len(self.fan_controller.fan_paths),
            'cpu_control_available': self.cpu_controller.is_available
        }

def load_config(config_path: str = "/etc/thermal_guardian.conf") -> ThermalConfig:
    """Load configuration from file"""
    config = ThermalConfig()
    
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            logging.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            
    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Thermal Guardian - Advanced Thermal Management")
    parser.add_argument("--config", "-c", default="/etc/thermal_guardian.conf",
                       help="Configuration file path")
    parser.add_argument("--daemon", "-d", action="store_true",
                       help="Run as daemon")
    parser.add_argument("--status", "-s", action="store_true",
                       help="Show current status")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Test mode - run once and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.verbose:
        config.log_level = "DEBUG"
        
    # Initialize thermal manager
    thermal_manager = ThermalManager(config)
    
    if args.status:
        # Show status and exit
        status = thermal_manager.get_status()
        print(json.dumps(status, indent=2))
        return 0
        
    elif args.test:
        # Run one cycle and exit
        thermal_manager.run_monitoring_cycle()
        status = thermal_manager.get_status()
        print(f"Test complete. Current state: {status['current_state']}, "
              f"Max temp: {status['max_temperature']:.1f}°C")
        return 0
        
    else:
        # Run normal monitoring
        if args.daemon:
            # TODO: Implement proper daemonization
            pass
            
        thermal_manager.run()
        return 0

if __name__ == "__main__":
    sys.exit(main())
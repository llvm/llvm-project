#!/usr/bin/env python3
"""
Thermal Guardian System v3.0 for Dell LAT5150DRVMIL - Enhanced Edition
=====================================================================

Advanced thermal protection system with neural prediction, adaptive control,
and military-grade reliability features.

Key Enhancements:
- Neural network-based temperature prediction
- Adaptive PID control with self-tuning
- Advanced sensor fusion with Kalman filtering
- Redundant control pathways
- Performance metrics and telemetry
- Self-diagnostic capabilities
- Hardware abstraction layer for portability

Critical Thresholds:
- 103°C: CoreTemp emergency throttling  
- 105°C: Hardware emergency shutdown
- Operating range: 90-105°C with graduated response

Author: Thermal Guardian Agent Team
Version: 3.0 (Enhanced Edition)
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
import fcntl
import tempfile
import hashlib
import struct
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from collections import deque
from enum import IntEnum
import math
import statistics
import numpy as np

# Security constants for input validation
TEMP_MIN = -40000  # -40°C in millidegrees
TEMP_MAX = 150000  # 150°C in millidegrees
PWM_MIN = 0        # Minimum PWM value
PWM_MAX = 255      # Maximum PWM value
PERF_MIN = 1       # Minimum performance percentage
PERF_MAX = 100     # Maximum performance percentage

# Process isolation
LOCKFILE_PATH = '/var/run/thermal-guardian.lock'
METRICS_PATH = '/var/lib/thermal-guardian/metrics.json'
STATE_PATH = '/var/lib/thermal-guardian/state.json'

# Hardware abstraction layer
class HardwareInterface:
    """Abstract hardware interface for portability"""
    
    # Thermal sensor paths (updated based on actual system)
    THERMAL_SENSORS = {
        'x86_pkg_temp': '/sys/class/thermal/thermal_zone9/temp',
        'dell_tcpu': '/sys/class/thermal/thermal_zone7/temp',
        'coretemp': '/sys/class/hwmon/hwmon7/temp1_input',
        'dell_cpu': '/sys/class/hwmon/hwmon5/temp1_input',
        'dell_smm': '/sys/class/hwmon/hwmon6/temp1_input'
    }
    
    # Control interfaces
    FAN_CONTROL = '/sys/class/hwmon/hwmon6/pwm1'
    FAN_MAX = '/sys/class/hwmon/hwmon6/pwm1_max'
    CPU_FREQ_CONTROL = '/sys/devices/system/cpu/intel_pstate/max_perf_pct'
    TURBO_CONTROL = '/sys/devices/system/cpu/intel_pstate/no_turbo'
    
    @classmethod
    def get_sensor_paths(cls) -> Dict[str, str]:
        """Get thermal sensor paths with auto-discovery"""
        sensors = cls.THERMAL_SENSORS.copy()
        
        # Auto-discover additional sensors
        hwmon_base = Path('/sys/class/hwmon')
        if hwmon_base.exists():
            for hwmon_dir in hwmon_base.iterdir():
                name_file = hwmon_dir / 'name'
                if name_file.exists():
                    try:
                        driver_name = name_file.read_text().strip()
                        # Look for temperature inputs
                        for temp_file in hwmon_dir.glob('temp*_input'):
                            sensor_name = f"{driver_name}_{temp_file.stem}"
                            if sensor_name not in sensors:
                                sensors[sensor_name] = str(temp_file)
                    except Exception:
                        pass
        
        return sensors

class ThermalPhase(IntEnum):
    """Thermal management phases"""
    NORMAL = 0
    PREVENTIVE = 1
    ACTIVE = 2
    AGGRESSIVE = 3
    MAXIMUM = 4
    EMERGENCY = 5

@dataclass
class ThermalReading:
    """Single thermal sensor reading with metadata"""
    temperature: float
    timestamp: float
    sensor_name: str
    valid: bool = True
    confidence: float = 1.0
    raw_value: Optional[int] = None

@dataclass
class ThermalState:
    """Enhanced thermal system state"""
    composite_temp: float
    predicted_temp: float
    confidence: float
    current_phase: ThermalPhase
    fan_speed: int
    cpu_limit: int
    turbo_enabled: bool
    timestamp: float
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    prediction_error: float = 0.0
    control_effort: float = 0.0

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    uptime: float = 0.0
    total_readings: int = 0
    failed_readings: int = 0
    phase_transitions: int = 0
    emergency_events: int = 0
    max_temperature: float = 0.0
    avg_temperature: float = 0.0
    prediction_accuracy: float = 0.0
    control_efficiency: float = 0.0
    last_update: float = 0.0

class KalmanFilter:
    """Kalman filter for sensor fusion"""
    
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = None
        self.posteri_error_estimate = 1.0
        
    def update(self, measurement: float) -> float:
        """Update filter with new measurement"""
        if self.posteri_estimate is None:
            self.posteri_estimate = measurement
            return measurement
            
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        
        return self.posteri_estimate

class NeuralPredictor:
    """Simple neural network for temperature prediction"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)
        
        # Adam optimizer parameters
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.t = 0
        
    def predict(self, x: np.ndarray) -> float:
        """Forward pass"""
        # Hidden layer with ReLU
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)
        
        # Output layer
        z2 = np.dot(a1, self.w2) + self.b2
        
        return float(z2[0])
        
    def train(self, x: np.ndarray, y: float, learning_rate: float = 0.001):
        """Train the network with one sample"""
        self.t += 1
        
        # Forward pass
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.w2) + self.b2
        prediction = z2[0]
        
        # Backward pass
        error = prediction - y
        
        # Output layer gradients
        dz2 = error
        dw2 = np.dot(a1.reshape(-1, 1), dz2.reshape(1, -1))
        db2 = dz2
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (z1 > 0)
        dw1 = np.dot(x.reshape(-1, 1), dz1.reshape(1, -1))
        db1 = dz1
        
        # Adam optimizer update
        # W1
        self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * dw1
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (dw1 ** 2)
        m_hat_w1 = self.m_w1 / (1 - self.beta1 ** self.t)
        v_hat_w1 = self.v_w1 / (1 - self.beta2 ** self.t)
        self.w1 -= learning_rate * m_hat_w1 / (np.sqrt(v_hat_w1) + self.epsilon)
        
        # W2
        self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * dw2
        self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (dw2 ** 2)
        m_hat_w2 = self.m_w2 / (1 - self.beta1 ** self.t)
        v_hat_w2 = self.v_w2 / (1 - self.beta2 ** self.t)
        self.w2 -= learning_rate * m_hat_w2 / (np.sqrt(v_hat_w2) + self.epsilon)
        
        # Biases (simple gradient descent for stability)
        self.b1 -= learning_rate * db1
        self.b2 -= learning_rate * db2

class PIDController:
    """PID controller with anti-windup and derivative filtering"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01,
                 setpoint: float = 85.0, output_limits: Tuple[float, float] = (0, 255)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.last_output = 0.0
        
        # Anti-windup
        self.integral_limit = 100.0
        
        # Derivative filter
        self.derivative_filter = 0.0
        self.filter_coefficient = 0.1
        
    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """Calculate PID output"""
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.last_error = self.setpoint - measurement
            return self.output_limits[0]
            
        if dt is None:
            dt = current_time - self.last_time
            
        if dt <= 0:
            return self.last_output
            
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.error_sum
        
        # Derivative term with filtering
        derivative = (error - self.last_error) / dt
        self.derivative_filter = (self.filter_coefficient * derivative + 
                                 (1 - self.filter_coefficient) * self.derivative_filter)
        d_term = self.kd * self.derivative_filter
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply limits
        output = np.clip(output, *self.output_limits)
        
        # Anti-windup: if output is saturated, stop integrating
        if output >= self.output_limits[1] or output <= self.output_limits[0]:
            self.error_sum -= error * dt
            
        # Update state
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        
        return output
        
    def reset(self):
        """Reset controller state"""
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.derivative_filter = 0.0

class AdvancedThermalSensor:
    """Enhanced thermal sensor with Kalman filtering and fault detection"""
    
    def __init__(self, name: str, path: str, weight: float, reliability: float):
        self.name = name
        self.path = self._validate_path(path)
        self.weight = weight
        self.reliability = reliability
        self.last_reading = None
        self.history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.kalman_filter = KalmanFilter()
        self.fault_counter = 0
        self.max_faults = 5
        self._file_descriptor = None
        
    def _validate_path(self, path: str) -> str:
        """Validate and canonicalize sensor path"""
        try:
            canonical_path = os.path.realpath(path)
            allowed_bases = ['/sys/class/thermal', '/sys/class/hwmon', '/sys/devices']
            if not any(canonical_path.startswith(base) for base in allowed_bases):
                raise ValueError(f"Sensor path outside allowed locations: {canonical_path}")
            return canonical_path
        except Exception as e:
            logging.error(f"Invalid sensor path {path}: {e}")
            raise
    
    def _atomic_read(self, file_path: str) -> Optional[str]:
        """Perform atomic file read with retry logic"""
        max_retries = 3
        retry_delay = 0.01  # 10ms
        
        for attempt in range(max_retries):
            try:
                with open(file_path, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    content = f.read().strip()
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return content
            except (IOError, OSError, BlockingIOError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.debug(f"Atomic read failed for {file_path}: {e}")
                    return None
        return None
        
    def read_temperature(self) -> Optional[ThermalReading]:
        """Read temperature with enhanced validation and filtering"""
        try:
            if not os.path.exists(self.path):
                self.fault_counter += 1
                return None
                
            temp_raw = self._atomic_read(self.path)
            if temp_raw is None:
                self.fault_counter += 1
                return None
                
            # Validate input
            if not temp_raw.lstrip('-').isdigit():
                logging.warning(f"Non-numeric temperature reading from {self.name}: {temp_raw}")
                self.fault_counter += 1
                return None
                
            temp_millidegrees = int(temp_raw)
            
            # Bounds checking
            if not (TEMP_MIN <= temp_millidegrees <= TEMP_MAX):
                logging.warning(
                    f"Temperature out of valid range from {self.name}: "
                    f"{temp_millidegrees} (valid: {TEMP_MIN} to {TEMP_MAX})"
                )
                self.fault_counter += 1
                return None
                
            # Convert to celsius
            temp_celsius = temp_millidegrees / 1000.0
            
            # Apply Kalman filtering
            filtered_temp = self.kalman_filter.update(temp_celsius)
            
            # Calculate confidence based on deviation from filtered value
            deviation = abs(temp_celsius - filtered_temp)
            confidence = math.exp(-deviation / 5.0)  # 5°C deviation = ~37% confidence
            
            reading = ThermalReading(
                temperature=filtered_temp,
                timestamp=time.time(),
                sensor_name=self.name,
                valid=True,
                confidence=confidence,
                raw_value=temp_millidegrees
            )
            
            # Reset fault counter on successful read
            self.fault_counter = 0
            
            self.last_reading = reading
            self.history.append(reading)
            return reading
            
        except Exception as e:
            logging.error(f"Unexpected error reading {self.name}: {e}")
            self.fault_counter += 1
            return None
    
    def is_healthy(self) -> bool:
        """Check if sensor is healthy"""
        return self.fault_counter < self.max_faults
    
    def get_statistics(self) -> Dict:
        """Get sensor statistics"""
        if len(self.history) < 2:
            return {}
            
        temps = [r.temperature for r in self.history]
        return {
            'mean': statistics.mean(temps),
            'stdev': statistics.stdev(temps) if len(temps) > 1 else 0,
            'min': min(temps),
            'max': max(temps),
            'trend': self.get_temperature_rate(),
            'health': self.is_healthy()
        }
    
    def get_temperature_rate(self) -> float:
        """Calculate temperature change rate using linear regression"""
        if len(self.history) < 5:
            return 0.0
            
        recent = list(self.history)[-30:]  # Last 30 seconds
        n = len(recent)
        
        if n < 2:
            return 0.0
            
        # Use numpy for efficient linear regression
        x = np.arange(n)
        y = np.array([r.temperature for r in recent])
        
        # Calculate slope using least squares
        A = np.vstack([x, np.ones(n)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return slope  # °C per second

class ThermalGuardianV3:
    """Enhanced Thermal Guardian with advanced features"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Process isolation
        self._lockfile = None
        self._acquire_lock()
        
        try:
            # Initialize components
            self.config = self._load_config(config_path)
            self.hardware = HardwareInterface()
            self.sensors = self._initialize_sensors()
            self.current_phase = ThermalPhase.NORMAL
            self.phase_entry_time = time.time()
            self.running = False
            self.thermal_history = deque(maxlen=600)  # 10 minutes
            
            # Advanced components
            self.neural_predictor = NeuralPredictor()
            self.fan_pid = PIDController(
                kp=2.5, ki=0.5, kd=0.1,
                setpoint=85.0,
                output_limits=(0, 255)
            )
            self.cpu_pid = PIDController(
                kp=-1.5, ki=-0.3, kd=-0.05,
                setpoint=90.0,
                output_limits=(30, 100)
            )
            
            # Metrics and diagnostics
            self.metrics = PerformanceMetrics()
            self.diagnostics = {}
            
            # Setup
            self._setup_logging()
            self._setup_directories()
            self._load_state()
            
            # Phase definitions
            self.phases = self._setup_phases()
            
            logging.info("Thermal Guardian v3.0 initialized successfully")
            self._log_system_info()
            
        except Exception as e:
            self._release_lock()
            raise RuntimeError(f"Failed to initialize Thermal Guardian: {e}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            Path('/var/lib/thermal-guardian'),
            Path('/var/log/thermal-guardian')
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _log_system_info(self):
        """Log system information"""
        try:
            # CPU info
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_model = line.split(':')[1].strip()
                        logging.info(f"CPU Model: {cpu_model}")
                        break
            
            # Available sensors
            logging.info(f"Available sensors: {list(self.sensors.keys())}")
            
            # System limits
            if os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'):
                with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq', 'r') as f:
                    max_freq = int(f.read()) / 1000  # Convert to MHz
                    logging.info(f"CPU Max Frequency: {max_freq:.0f} MHz")
                    
        except Exception as e:
            logging.debug(f"Failed to get system info: {e}")
    
    def _setup_phases(self) -> Dict[ThermalPhase, Dict]:
        """Setup thermal phases with validation"""
        return {
            ThermalPhase.NORMAL: {
                'name': 'Normal Operations',
                'enter': 0,
                'exit': 0,
                'color': '\033[92m',  # Green
                'icon': '✅'
            },
            ThermalPhase.PREVENTIVE: {
                'name': 'Preventive Cooling',
                'enter': 85,
                'exit': 83,
                'color': '\033[93m',  # Yellow
                'icon': '🟡'
            },
            ThermalPhase.ACTIVE: {
                'name': 'Active Management',
                'enter': 90,
                'exit': 87,
                'color': '\033[93m',  # Yellow
                'icon': '⚠️'
            },
            ThermalPhase.AGGRESSIVE: {
                'name': 'Aggressive Cooling',
                'enter': 95,
                'exit': 91,
                'color': '\033[91m',  # Red
                'icon': '🔥'
            },
            ThermalPhase.MAXIMUM: {
                'name': 'Maximum Protection',
                'enter': 100,
                'exit': 96,
                'color': '\033[91m',  # Red
                'icon': '🚨'
            },
            ThermalPhase.EMERGENCY: {
                'name': 'EMERGENCY MODE',
                'enter': 103,
                'exit': 99,
                'color': '\033[91m',  # Red
                'icon': '💀'
            }
        }
    
    def _acquire_lock(self):
        """Acquire process lock with improved error handling"""
        try:
            # Ensure directory exists
            Path(LOCKFILE_PATH).parent.mkdir(parents=True, exist_ok=True)
            
            self._lockfile = open(LOCKFILE_PATH, 'w')
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lockfile.write(f"{os.getpid()}\n")
            self._lockfile.write(f"{time.time()}\n")
            self._lockfile.flush()
            
        except (IOError, BlockingIOError):
            if self._lockfile:
                self._lockfile.close()
            
            # Check if old process is still running
            if self._check_stale_lock():
                # Retry after removing stale lock
                os.unlink(LOCKFILE_PATH)
                self._acquire_lock()
            else:
                raise RuntimeError(f"Another instance is already running (lockfile: {LOCKFILE_PATH})")
    
    def _check_stale_lock(self) -> bool:
        """Check if lockfile is stale"""
        try:
            with open(LOCKFILE_PATH, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    pid = int(lines[0].strip())
                    timestamp = float(lines[1].strip())
                    
                    # Check if process exists
                    try:
                        os.kill(pid, 0)
                        return False  # Process exists
                    except OSError:
                        # Process doesn't exist
                        if time.time() - timestamp > 300:  # 5 minutes
                            logging.warning(f"Removing stale lockfile (PID {pid})")
                            return True
        except Exception:
            pass
        return False
    
    def _release_lock(self):
        """Release process lock"""
        if self._lockfile:
            try:
                fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_UN)
                self._lockfile.close()
                os.unlink(LOCKFILE_PATH)
            except Exception as e:
                logging.warning(f"Failed to release lock: {e}")
            finally:
                self._lockfile = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load and validate configuration"""
        default_config = {
            'monitoring_interval': 1.0,
            'prediction_horizon': 10.0,
            'sensor_weights': {
                'x86_pkg_temp': 0.35,
                'dell_tcpu': 0.30,
                'coretemp': 0.25,
                'dell_cpu': 0.05,
                'dell_smm': 0.05
            },
            'phase_delays': {
                1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 0.0
            },
            'max_fan_pwm': 255,
            'emergency_temp': 105.0,
            'critical_temp': 103.0,
            'target_temp': 85.0,
            'enable_neural_prediction': True,
            'enable_adaptive_pid': True,
            'telemetry_interval': 60.0,
            'max_log_size_mb': 100
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    
                # Validate user config
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(default_config[key], (int, float)):
                            if not isinstance(value, (int, float)):
                                logging.warning(f"Invalid type for {key}, using default")
                                continue
                        default_config[key] = value
                    else:
                        logging.warning(f"Unknown config key: {key}")
                        
                logging.info(f"Loaded config from {config_path}")
            except Exception as e:
                logging.warning(f"Failed to load config: {e}, using defaults")
                
        return default_config
    
    def _initialize_sensors(self) -> Dict[str, AdvancedThermalSensor]:
        """Initialize thermal sensors with auto-discovery"""
        sensors = {}
        sensor_paths = self.hardware.get_sensor_paths()
        
        for name, path in sensor_paths.items():
            weight = self.config['sensor_weights'].get(name, 0.0)
            
            # Skip sensors with zero weight unless no other sensors available
            if weight == 0.0 and len(sensors) > 0:
                continue
                
            reliability = 0.9 if name in ['x86_pkg_temp', 'coretemp'] else 0.8
            
            sensor = AdvancedThermalSensor(name, path, weight, reliability)
            
            # Test sensor
            reading = sensor.read_temperature()
            if reading:
                sensors[name] = sensor
                logging.info(f"Initialized sensor {name}: {reading.temperature:.1f}°C (filtered)")
            else:
                logging.warning(f"Failed to initialize sensor {name}")
                
        if not sensors:
            raise RuntimeError("No thermal sensors available")
            
        # Normalize weights
        total_weight = sum(s.weight for s in sensors.values())
        if total_weight > 0:
            for sensor in sensors.values():
                sensor.weight /= total_weight
                
        return sensors
    
    def _setup_logging(self):
        """Setup advanced logging with rotation"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Check log size and rotate if needed
        log_file = Path('/var/log/thermal-guardian/thermal_guardian.log')
        if log_file.exists() and log_file.stat().st_size > self.config['max_log_size_mb'] * 1024 * 1024:
            # Rotate log
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_file.rename(log_file.with_suffix(f'.{timestamp}.log'))
            
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _save_state(self):
        """Save current state for recovery"""
        try:
            state = {
                'timestamp': time.time(),
                'phase': self.current_phase.value,
                'metrics': {
                    'uptime': self.metrics.uptime,
                    'total_readings': self.metrics.total_readings,
                    'failed_readings': self.metrics.failed_readings,
                    'phase_transitions': self.metrics.phase_transitions,
                    'emergency_events': self.metrics.emergency_events,
                    'max_temperature': self.metrics.max_temperature,
                    'avg_temperature': self.metrics.avg_temperature,
                    'prediction_accuracy': self.metrics.prediction_accuracy,
                    'control_efficiency': self.metrics.control_efficiency
                },
                'neural_weights': {
                    'w1': self.neural_predictor.w1.tolist(),
                    'b1': self.neural_predictor.b1.tolist(),
                    'w2': self.neural_predictor.w2.tolist(),
                    'b2': self.neural_predictor.b2.tolist()
                }
            }
            
            # Atomic write
            temp_file = Path(STATE_PATH + '.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            temp_file.replace(STATE_PATH)
            
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load saved state"""
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, 'r') as f:
                    state = json.load(f)
                    
                # Restore metrics
                if 'metrics' in state:
                    for key, value in state['metrics'].items():
                        if hasattr(self.metrics, key):
                            setattr(self.metrics, key, value)
                            
                # Restore neural network weights
                if 'neural_weights' in state and self.config['enable_neural_prediction']:
                    try:
                        self.neural_predictor.w1 = np.array(state['neural_weights']['w1'])
                        self.neural_predictor.b1 = np.array(state['neural_weights']['b1'])
                        self.neural_predictor.w2 = np.array(state['neural_weights']['w2'])
                        self.neural_predictor.b2 = np.array(state['neural_weights']['b2'])
                        logging.info("Restored neural network weights")
                    except Exception as e:
                        logging.warning(f"Failed to restore neural weights: {e}")
                        
                logging.info("Loaded previous state")
                
        except Exception as e:
            logging.debug(f"No previous state to load: {e}")
    
    def get_composite_temperature(self) -> Tuple[float, float, int]:
        """Get composite temperature with confidence score"""
        readings = {}
        weights = {}
        valid_sensors = 0
        
        # Read all sensors
        for name, sensor in self.sensors.items():
            if not sensor.is_healthy():
                logging.warning(f"Sensor {name} is unhealthy, skipping")
                continue
                
            reading = sensor.read_temperature()
            if reading and reading.valid:
                readings[name] = reading
                # Adjust weight based on confidence
                weights[name] = sensor.weight * reading.confidence
                valid_sensors += 1
        
        if not readings:
            logging.error("No valid sensor readings!")
            # Return last known temperature if available
            if self.thermal_history:
                last_state = self.thermal_history[-1]
                return last_state.composite_temp, 0.0, 0
            return 0.0, 0.0, 0
        
        # Weighted average
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Simple average fallback
            composite_temp = sum(r.temperature for r in readings.values()) / len(readings)
            confidence = 0.5
        else:
            weighted_sum = sum(r.temperature * weights[name] 
                             for name, r in readings.items())
            composite_temp = weighted_sum / total_weight
            
            # Calculate confidence based on sensor agreement
            temps = [r.temperature for r in readings.values()]
            if len(temps) > 1:
                temp_stdev = statistics.stdev(temps)
                confidence = math.exp(-temp_stdev / 3.0)  # 3°C stdev = ~37% confidence
            else:
                confidence = readings[list(readings.keys())[0]].confidence
        
        # Sanity check against physical limits
        if composite_temp > 110:
            logging.warning(f"Composite temperature {composite_temp}°C exceeds physical limits")
            composite_temp = 110.0
            
        return composite_temp, confidence, valid_sensors
    
    def predict_temperature_neural(self, horizon_seconds: float = 10.0) -> float:
        """Predict temperature using neural network"""
        if len(self.thermal_history) < 10:
            # Not enough data for neural prediction
            return self.predict_temperature_simple(horizon_seconds)
        
        # Prepare input features
        recent_states = list(self.thermal_history)[-10:]
        features = []
        
        # Temperature values
        for state in recent_states:
            features.append(state.composite_temp / 100.0)  # Normalize
            
        # Add time features
        current_time = time.time()
        time_since_start = (current_time - self.phase_entry_time) / 300.0  # Normalize to 5 min
        features.append(time_since_start)
        
        # Current control state
        if recent_states:
            features.append(recent_states[-1].fan_speed / 255.0)
            features.append(recent_states[-1].cpu_limit / 100.0)
        
        # Pad if needed
        while len(features) < self.neural_predictor.input_size:
            features.append(0.0)
            
        features = np.array(features[:self.neural_predictor.input_size])
        
        # Predict
        prediction = self.neural_predictor.predict(features) * 100.0  # Denormalize
        
        # Sanity check
        current_temp = recent_states[-1].composite_temp if recent_states else 50.0
        max_change = horizon_seconds * 2.0  # Max 2°C/s
        
        prediction = np.clip(prediction, 
                           current_temp - max_change,
                           current_temp + max_change)
        
        return float(prediction)
    
    def predict_temperature_simple(self, horizon_seconds: float = 10.0) -> float:
        """Simple temperature prediction using trend analysis"""
        if len(self.thermal_history) < 2:
            current_temp, _, _ = self.get_composite_temperature()
            return current_temp
        
        # Linear regression on recent data
        recent_states = list(self.thermal_history)[-30:]
        
        if len(recent_states) < 2:
            return recent_states[-1].composite_temp
            
        # Extract times and temperatures
        times = np.array([s.timestamp for s in recent_states])
        temps = np.array([s.composite_temp for s in recent_states])
        
        # Normalize time
        times = times - times[0]
        
        # Fit linear model
        A = np.vstack([times, np.ones(len(times))]).T
        slope, intercept = np.linalg.lstsq(A, temps, rcond=None)[0]
        
        # Predict
        future_time = times[-1] + horizon_seconds
        prediction = slope * future_time + intercept
        
        # Apply thermal inertia model
        thermal_inertia = 20.0  # seconds
        inertia_factor = 1.0 - math.exp(-horizon_seconds / thermal_inertia)
        
        current_temp = temps[-1]
        prediction = current_temp + (prediction - current_temp) * inertia_factor
        
        return float(prediction)
    
    def determine_thermal_phase(self) -> ThermalPhase:
        """Determine thermal phase with hysteresis and prediction"""
        current_temp, confidence, _ = self.get_composite_temperature()
        
        # Use neural or simple prediction based on config
        if self.config['enable_neural_prediction']:
            predicted_temp = self.predict_temperature_neural(10.0)
        else:
            predicted_temp = self.predict_temperature_simple(10.0)
        
        # Weight prediction based on confidence
        effective_temp = current_temp * (0.5 + 0.5 * confidence) + predicted_temp * (0.5 - 0.5 * confidence)
        
        # Phase escalation
        for phase in [ThermalPhase.EMERGENCY, ThermalPhase.MAXIMUM, 
                     ThermalPhase.AGGRESSIVE, ThermalPhase.ACTIVE, 
                     ThermalPhase.PREVENTIVE]:
            if effective_temp >= self.phases[phase]['enter']:
                if self.current_phase < phase:
                    # Check delay requirement
                    delay = self.config['phase_delays'].get(phase.value, 0.0)
                    if delay == 0.0 or self._temperature_sustained_above(
                        self.phases[phase]['enter'], delay):
                        self._enter_phase(phase)
                        return phase
        
        # Phase de-escalation
        if self.current_phase > ThermalPhase.NORMAL:
            time_in_phase = time.time() - self.phase_entry_time
            exit_temp = self.phases[self.current_phase]['exit']
            
            # Require longer time for de-escalation
            min_time = 5.0 if self.current_phase >= ThermalPhase.AGGRESSIVE else 3.0
            
            if effective_temp <= exit_temp and time_in_phase > min_time:
                self._exit_to_lower_phase()
        
        return self.current_phase
    
    def _temperature_sustained_above(self, threshold: float, duration: float) -> bool:
        """Check if temperature sustained above threshold"""
        if len(self.thermal_history) < 2:
            return False
            
        cutoff_time = time.time() - duration
        sustained_readings = [
            state for state in self.thermal_history 
            if state.timestamp >= cutoff_time and state.composite_temp >= threshold
        ]
        
        required_readings = max(1, int(duration / self.config['monitoring_interval']))
        return len(sustained_readings) >= required_readings * 0.8  # 80% threshold
    
    def _enter_phase(self, phase: ThermalPhase):
        """Enter new thermal phase"""
        old_phase = self.current_phase
        self.current_phase = phase
        self.phase_entry_time = time.time()
        self.metrics.phase_transitions += 1
        
        phase_info = self.phases[phase]
        logging.warning(
            f"{phase_info['color']}Thermal phase escalation: "
            f"{self.phases[old_phase]['name']} → {phase_info['name']} "
            f"{phase_info['icon']}\033[0m"
        )
        
        # Reset PID controllers on phase change
        if self.config['enable_adaptive_pid']:
            self._adapt_pid_parameters(phase)
    
    def _exit_to_lower_phase(self):
        """Exit to lower thermal phase"""
        old_phase = self.current_phase
        current_temp, _, _ = self.get_composite_temperature()
        
        # Find appropriate lower phase
        new_phase = ThermalPhase.NORMAL
        for phase in [ThermalPhase.MAXIMUM, ThermalPhase.AGGRESSIVE,
                     ThermalPhase.ACTIVE, ThermalPhase.PREVENTIVE]:
            if phase < old_phase and current_temp >= self.phases[phase]['exit']:
                new_phase = phase
                break
        
        self.current_phase = new_phase
        self.phase_entry_time = time.time()
        self.metrics.phase_transitions += 1
        
        phase_info = self.phases[new_phase]
        logging.info(
            f"{phase_info['color']}Thermal phase de-escalation: "
            f"{self.phases[old_phase]['name']} → {phase_info['name']} "
            f"{phase_info['icon']}\033[0m"
        )
        
        # Adapt PID on phase change
        if self.config['enable_adaptive_pid']:
            self._adapt_pid_parameters(new_phase)
    
    def _adapt_pid_parameters(self, phase: ThermalPhase):
        """Adapt PID parameters based on phase"""
        # Fan PID tuning
        if phase == ThermalPhase.NORMAL:
            self.fan_pid.kp = 1.5
            self.fan_pid.ki = 0.3
            self.fan_pid.kd = 0.05
        elif phase == ThermalPhase.PREVENTIVE:
            self.fan_pid.kp = 2.0
            self.fan_pid.ki = 0.4
            self.fan_pid.kd = 0.08
        elif phase == ThermalPhase.ACTIVE:
            self.fan_pid.kp = 3.0
            self.fan_pid.ki = 0.6
            self.fan_pid.kd = 0.1
        elif phase >= ThermalPhase.AGGRESSIVE:
            self.fan_pid.kp = 4.0
            self.fan_pid.ki = 0.8
            self.fan_pid.kd = 0.15
            
        # CPU PID tuning (negative gains for inverse control)
        if phase <= ThermalPhase.PREVENTIVE:
            self.cpu_pid.kp = -1.0
            self.cpu_pid.ki = -0.2
            self.cpu_pid.kd = -0.03
        elif phase == ThermalPhase.ACTIVE:
            self.cpu_pid.kp = -1.5
            self.cpu_pid.ki = -0.3
            self.cpu_pid.kd = -0.05
        else:
            self.cpu_pid.kp = -2.0
            self.cpu_pid.ki = -0.4
            self.cpu_pid.kd = -0.08
            
        # Reset integral terms
        self.fan_pid.error_sum = 0
        self.cpu_pid.error_sum = 0
    
    def execute_phase_actions(self, phase: ThermalPhase) -> Dict:
        """Execute thermal management with PID control"""
        current_temp, confidence, _ = self.get_composite_temperature()
        actions = {}
        
        try:
            if phase == ThermalPhase.NORMAL:
                # Baseline cooling
                actions['fan_pwm'] = 100
                actions['cpu_limit'] = 100
                actions['turbo'] = True
                
            elif phase == ThermalPhase.PREVENTIVE:
                # Start ramping up cooling
                if self.config['enable_adaptive_pid']:
                    self.fan_pid.setpoint = 83.0
                    actions['fan_pwm'] = int(self.fan_pid.update(current_temp))
                else:
                    actions['fan_pwm'] = 180
                actions['cpu_limit'] = 100
                actions['turbo'] = True
                
            elif phase == ThermalPhase.ACTIVE:
                # Active cooling with mild throttling
                if self.config['enable_adaptive_pid']:
                    self.fan_pid.setpoint = 87.0
                    self.cpu_pid.setpoint = 88.0
                    actions['fan_pwm'] = int(self.fan_pid.update(current_temp))
                    actions['cpu_limit'] = int(self.cpu_pid.update(current_temp))
                else:
                    actions['fan_pwm'] = 220
                    actions['cpu_limit'] = 95
                actions['turbo'] = False
                
            elif phase == ThermalPhase.AGGRESSIVE:
                # Aggressive management
                if self.config['enable_adaptive_pid']:
                    self.fan_pid.setpoint = 91.0
                    self.cpu_pid.setpoint = 92.0
                    actions['fan_pwm'] = int(self.fan_pid.update(current_temp))
                    actions['cpu_limit'] = int(self.cpu_pid.update(current_temp))
                else:
                    actions['fan_pwm'] = 255
                    actions['cpu_limit'] = 80
                actions['turbo'] = False
                
            elif phase == ThermalPhase.MAXIMUM:
                # Maximum cooling, significant throttling
                actions['fan_pwm'] = 255
                if self.config['enable_adaptive_pid']:
                    self.cpu_pid.setpoint = 96.0
                    actions['cpu_limit'] = int(self.cpu_pid.update(current_temp))
                else:
                    actions['cpu_limit'] = 60
                actions['turbo'] = False
                
            elif phase == ThermalPhase.EMERGENCY:
                # Emergency mode - survival priority
                actions['fan_pwm'] = 255
                actions['cpu_limit'] = 40
                actions['turbo'] = False
                
                # Additional emergency actions
                self._execute_emergency_cooling()
            
            # Apply bounds checking
            actions['fan_pwm'] = np.clip(actions['fan_pwm'], PWM_MIN, PWM_MAX)
            actions['cpu_limit'] = np.clip(actions['cpu_limit'], 20, PERF_MAX)
            
            # Calculate control effort for metrics
            control_effort = (actions['fan_pwm'] / 255.0 * 0.7 + 
                            (100 - actions['cpu_limit']) / 100.0 * 0.3)
            
            # Apply actions
            self._apply_fan_control(actions['fan_pwm'])
            self._apply_cpu_control(actions['cpu_limit'])
            self._apply_turbo_control(actions['turbo'])
            
            # Update metrics
            self.metrics.control_efficiency = 1.0 - control_effort
            
        except Exception as e:
            logging.error(f"Failed to execute phase {phase} actions: {e}")
            # Fallback to safe defaults
            actions = {
                'fan_pwm': 200,
                'cpu_limit': 70,
                'turbo': False
            }
            self._apply_fan_control(actions['fan_pwm'])
            self._apply_cpu_control(actions['cpu_limit'])
            self._apply_turbo_control(actions['turbo'])
            
        return actions
    
    def _execute_emergency_cooling(self):
        """Execute additional emergency cooling measures"""
        try:
            # Kill non-essential processes
            high_cpu_procs = self._get_high_cpu_processes()
            for proc in high_cpu_procs[:3]:  # Kill top 3
                if proc['name'] not in ['systemd', 'kernel', 'thermal-guardian']:
                    logging.warning(f"Killing high-CPU process: {proc['name']} (PID: {proc['pid']})")
                    try:
                        os.kill(proc['pid'], signal.SIGTERM)
                    except Exception:
                        pass
                        
            # Sync and drop caches
            os.system("sync")
            os.system("echo 3 > /proc/sys/vm/drop_caches")
            
        except Exception as e:
            logging.error(f"Emergency cooling measures failed: {e}")
    
    def _get_high_cpu_processes(self) -> List[Dict]:
        """Get list of high CPU processes"""
        try:
            output = subprocess.check_output(
                ['ps', 'aux', '--sort=-%cpu'],
                universal_newlines=True
            )
            
            processes = []
            for line in output.strip().split('\n')[1:]:  # Skip header
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        'pid': int(parts[1]),
                        'cpu': float(parts[2]),
                        'name': parts[10]
                    })
                    
            return [p for p in processes if p['cpu'] > 10.0][:10]
            
        except Exception:
            return []
    
    def _atomic_write(self, file_path: str, value: str) -> bool:
        """Atomic write with verification"""
        try:
            canonical_path = os.path.realpath(file_path)
            allowed_bases = ['/sys/class/hwmon', '/sys/devices/system/cpu']
            if not any(canonical_path.startswith(base) for base in allowed_bases):
                logging.error(f"Write attempt to unauthorized path: {canonical_path}")
                return False
            
            # Write to temp file first
            temp_path = file_path + '.tmp'
            with open(temp_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(value)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            os.rename(temp_path, file_path)
            
            # Verify write
            with open(file_path, 'r') as f:
                if f.read().strip() != value.strip():
                    logging.warning(f"Write verification failed for {file_path}")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Failed to write {value} to {file_path}: {e}")
            # Cleanup temp file if exists
            try:
                os.unlink(file_path + '.tmp')
            except:
                pass
            return False
    
    def _apply_fan_control(self, pwm_value: int):
        """Apply fan control with validation"""
        if not isinstance(pwm_value, (int, np.integer)) or not (PWM_MIN <= pwm_value <= PWM_MAX):
            logging.error(f"Invalid PWM value: {pwm_value}")
            return
            
        success = self._atomic_write(self.hardware.FAN_CONTROL, str(int(pwm_value)))
        if success:
            logging.debug(f"Fan PWM set to {pwm_value}")
        else:
            logging.error(f"Failed to set fan PWM to {pwm_value}")
    
    def _apply_cpu_control(self, limit_percent: int):
        """Apply CPU frequency limit"""
        if not isinstance(limit_percent, (int, np.integer)) or not (PERF_MIN <= limit_percent <= PERF_MAX):
            logging.error(f"Invalid CPU limit: {limit_percent}%")
            return
            
        # Safety floor
        limit_percent = max(20, int(limit_percent))
        
        success = self._atomic_write(self.hardware.CPU_FREQ_CONTROL, str(limit_percent))
        if success:
            logging.debug(f"CPU frequency limit set to {limit_percent}%")
        else:
            logging.error(f"Failed to set CPU limit to {limit_percent}%")
    
    def _apply_turbo_control(self, enabled: bool):
        """Apply turbo boost control"""
        value = "0" if enabled else "1"  # Inverted logic
        
        success = self._atomic_write(self.hardware.TURBO_CONTROL, value)
        if success:
            status = "enabled" if enabled else "disabled"
            logging.debug(f"Turbo boost {status}")
        else:
            logging.error(f"Failed to set turbo boost")
    
    def check_emergency_conditions(self) -> bool:
        """Enhanced emergency condition detection"""
        current_temp, confidence, valid_sensors = self.get_composite_temperature()
        
        # No sensors available
        if valid_sensors == 0:
            logging.critical("EMERGENCY: No valid sensors!")
            return True
        
        # Any single sensor critical
        for sensor in self.sensors.values():
            if sensor.last_reading and sensor.last_reading.temperature > 104.0:
                logging.critical(
                    f"EMERGENCY: {sensor.name} at {sensor.last_reading.temperature:.1f}°C"
                )
                return True
        
        # Composite temperature critical
        if current_temp > self.config['emergency_temp']:
            logging.critical(f"EMERGENCY: Composite temperature {current_temp:.1f}°C")
            return True
        
        # Rapid temperature rise
        if len(self.thermal_history) >= 5:
            recent = list(self.thermal_history)[-5:]
            dt = recent[-1].timestamp - recent[0].timestamp
            if dt > 0:
                dtemp = recent[-1].composite_temp - recent[0].composite_temp
                rate = dtemp / dt
                if rate > 3.0:  # 3°C/s
                    logging.critical(f"EMERGENCY: Temperature rising at {rate:.1f}°C/s")
                    return True
        
        # Sustained high temperature
        high_temp_duration = sum(
            1 for state in list(self.thermal_history)[-60:]
            if state.composite_temp > 102
        )
        if high_temp_duration > 30:  # 30 seconds above 102°C
            logging.critical("EMERGENCY: Sustained critical temperature")
            return True
        
        return False
    
    def emergency_thermal_protection(self):
        """Enhanced emergency thermal protection"""
        logging.critical("EMERGENCY THERMAL PROTECTION ACTIVATED")
        self.metrics.emergency_events += 1
        
        try:
            # Maximum cooling
            self._apply_fan_control(255)
            
            # Severe throttling
            self._apply_cpu_control(25)
            self._apply_turbo_control(False)
            
            # Additional emergency measures
            self._execute_emergency_cooling()
            
            # Monitor for 5 seconds
            for i in range(5):
                time.sleep(1.0)
                current_temp, _, _ = self.get_composite_temperature()
                
                if current_temp < 100:
                    logging.info("Temperature recovering from emergency")
                    return
                    
            # Still critical after 5 seconds
            if current_temp > self.config['emergency_temp']:
                logging.critical("INITIATING EMERGENCY SHUTDOWN")
                self._save_state()  # Save state before shutdown
                
                # Notify user if possible
                try:
                    os.system("wall 'THERMAL EMERGENCY: System shutting down'")
                except:
                    pass
                    
                time.sleep(2.0)
                os.system("sudo shutdown -h now")
                
        except Exception as e:
            logging.critical(f"Emergency protection failed: {e}")
            # Last resort
            os.system("sudo shutdown -h now")
    
    def update_neural_predictor(self, actual_temp: float):
        """Update neural network with new data"""
        if not self.config['enable_neural_prediction']:
            return
            
        if len(self.thermal_history) < 11:
            return
            
        # Prepare training data
        history = list(self.thermal_history)[-11:-1]  # Previous 10 states
        features = []
        
        for state in history:
            features.append(state.composite_temp / 100.0)
            
        # Add time and control features
        time_feature = (time.time() - self.phase_entry_time) / 300.0
        features.append(time_feature)
        features.append(history[-1].fan_speed / 255.0)
        features.append(history[-1].cpu_limit / 100.0)
        
        # Pad
        while len(features) < self.neural_predictor.input_size:
            features.append(0.0)
            
        features = np.array(features[:self.neural_predictor.input_size])
        
        # Train
        target = actual_temp / 100.0
        self.neural_predictor.train(features, target, learning_rate=0.0001)
        
        # Update prediction accuracy metric
        prediction = self.neural_predictor.predict(features) * 100.0
        error = abs(prediction - actual_temp)
        
        # Exponential moving average
        alpha = 0.1
        self.metrics.prediction_accuracy = (
            (1 - alpha) * self.metrics.prediction_accuracy + 
            alpha * (1.0 - error / 10.0)  # 10°C error = 0% accuracy
        )
    
    def run_diagnostics(self) -> Dict:
        """Run system diagnostics"""
        diag = {
            'sensors': {},
            'controls': {},
            'predictions': {},
            'system': {}
        }
        
        # Sensor diagnostics
        for name, sensor in self.sensors.items():
            stats = sensor.get_statistics()
            diag['sensors'][name] = {
                'healthy': sensor.is_healthy(),
                'fault_count': sensor.fault_counter,
                'stats': stats
            }
        
        # Control diagnostics
        try:
            # Test fan control
            current_pwm = None
            if os.path.exists(self.hardware.FAN_CONTROL):
                with open(self.hardware.FAN_CONTROL, 'r') as f:
                    current_pwm = int(f.read().strip())
            diag['controls']['fan_operational'] = current_pwm is not None
            diag['controls']['fan_pwm'] = current_pwm
            
            # Test CPU control
            current_perf = None
            if os.path.exists(self.hardware.CPU_FREQ_CONTROL):
                with open(self.hardware.CPU_FREQ_CONTROL, 'r') as f:
                    current_perf = int(f.read().strip())
            diag['controls']['cpu_control_operational'] = current_perf is not None
            diag['controls']['cpu_limit'] = current_perf
            
        except Exception as e:
            logging.error(f"Control diagnostics failed: {e}")
        
        # Prediction diagnostics
        if self.config['enable_neural_prediction'] and len(self.thermal_history) > 10:
            current_temp, _, _ = self.get_composite_temperature()
            neural_pred = self.predict_temperature_neural(10.0)
            simple_pred = self.predict_temperature_simple(10.0)
            
            diag['predictions'] = {
                'current_temp': current_temp,
                'neural_prediction': neural_pred,
                'simple_prediction': simple_pred,
                'neural_error': abs(neural_pred - current_temp),
                'simple_error': abs(simple_pred - current_temp)
            }
        
        # System diagnostics
        diag['system'] = {
            'uptime': self.metrics.uptime,
            'phase': self.current_phase.name,
            'metrics': {
                'total_readings': self.metrics.total_readings,
                'failed_readings': self.metrics.failed_readings,
                'failure_rate': (self.metrics.failed_readings / 
                               max(1, self.metrics.total_readings)),
                'phase_transitions': self.metrics.phase_transitions,
                'emergency_events': self.metrics.emergency_events,
                'prediction_accuracy': self.metrics.prediction_accuracy,
                'control_efficiency': self.metrics.control_efficiency
            }
        }
        
        self.diagnostics = diag
        return diag
    
    def monitoring_loop(self):
        """Enhanced monitoring loop with telemetry"""
        logging.info("Starting enhanced monitoring loop")
        
        last_telemetry = 0
        last_save = 0
        
        while self.running:
            try:
                start_time = time.time()
                
                # Update metrics
                self.metrics.total_readings += 1
                self.metrics.uptime = time.time() - self.phase_entry_time
                
                # Get thermal state
                current_temp, confidence, valid_sensors = self.get_composite_temperature()
                
                if valid_sensors == 0:
                    self.metrics.failed_readings += 1
                    logging.error("No valid sensors - entering safe mode")
                    self._apply_fan_control(200)
                    self._apply_cpu_control(70)
                    time.sleep(1.0)
                    continue
                
                # Update temperature metrics
                if current_temp > self.metrics.max_temperature:
                    self.metrics.max_temperature = current_temp
                    
                # Update average (exponential moving average)
                alpha = 0.01
                self.metrics.avg_temperature = (
                    (1 - alpha) * self.metrics.avg_temperature +
                    alpha * current_temp
                )
                
                # Check emergency first
                if self.check_emergency_conditions():
                    self.emergency_thermal_protection()
                    continue
                
                # Determine phase
                phase = self.determine_thermal_phase()
                
                # Get predictions
                if self.config['enable_neural_prediction']:
                    predicted_temp = self.predict_temperature_neural()
                else:
                    predicted_temp = self.predict_temperature_simple()
                
                # Execute control actions
                actions = self.execute_phase_actions(phase)
                
                # Record state
                state = ThermalState(
                    composite_temp=current_temp,
                    predicted_temp=predicted_temp,
                    confidence=confidence,
                    current_phase=phase,
                    fan_speed=actions.get('fan_pwm', 0),
                    cpu_limit=actions.get('cpu_limit', 100),
                    turbo_enabled=actions.get('turbo', True),
                    timestamp=time.time(),
                    sensor_readings={name: s.last_reading.temperature 
                                   for name, s in self.sensors.items() 
                                   if s.last_reading},
                    prediction_error=abs(predicted_temp - current_temp),
                    control_effort=(actions.get('fan_pwm', 0) / 255.0 * 0.7 +
                                  (100 - actions.get('cpu_limit', 100)) / 100.0 * 0.3)
                )
                
                self.thermal_history.append(state)
                
                # Update neural network
                if len(self.thermal_history) > 11:
                    self.update_neural_predictor(current_temp)
                
                # Log status
                if phase != ThermalPhase.NORMAL or current_temp > 80:
                    phase_info = self.phases[phase]
                    logging.info(
                        f"{phase_info['color']}Thermal: {current_temp:.1f}°C "
                        f"(pred: {predicted_temp:.1f}°C, conf: {confidence:.2f}) "
                        f"Phase: {phase_info['name']} {phase_info['icon']} "
                        f"Fan: {actions.get('fan_pwm', 0)} "
                        f"CPU: {actions.get('cpu_limit', 100)}%\033[0m"
                    )
                
                # Periodic telemetry
                if time.time() - last_telemetry > self.config['telemetry_interval']:
                    self.run_diagnostics()
                    self._save_metrics()
                    last_telemetry = time.time()
                
                # Periodic state save
                if time.time() - last_save > 300:  # 5 minutes
                    self._save_state()
                    last_save = time.time()
                
                # Sleep for monitoring interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config['monitoring_interval'] - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}", exc_info=True)
                self.metrics.failed_readings += 1
                time.sleep(1.0)
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_data = {
                'timestamp': time.time(),
                'uptime': self.metrics.uptime,
                'readings': {
                    'total': self.metrics.total_readings,
                    'failed': self.metrics.failed_readings,
                    'success_rate': 1.0 - (self.metrics.failed_readings / 
                                         max(1, self.metrics.total_readings))
                },
                'temperature': {
                    'current': self.thermal_history[-1].composite_temp if self.thermal_history else 0,
                    'average': self.metrics.avg_temperature,
                    'maximum': self.metrics.max_temperature
                },
                'control': {
                    'phase_transitions': self.metrics.phase_transitions,
                    'emergency_events': self.metrics.emergency_events,
                    'prediction_accuracy': self.metrics.prediction_accuracy,
                    'control_efficiency': self.metrics.control_efficiency
                },
                'diagnostics': self.diagnostics
            }
            
            # Atomic write
            temp_file = Path(METRICS_PATH + '.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            temp_file.replace(METRICS_PATH)
            
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")
    
    def start(self):
        """Start thermal guardian"""
        logging.info("Starting Thermal Guardian v3.0")
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)
        
        try:
            self.monitoring_loop()
        finally:
            self.stop()
    
    def stop(self):
        """Enhanced shutdown with cleanup"""
        logging.info("Initiating graceful shutdown...")
        self.running = False
        
        # Save final state
        try:
            self._save_state()
            self._save_metrics()
            logging.info("Saved final state and metrics")
        except Exception as e:
            logging.error(f"Failed to save final state: {e}")
        
        # Restore safe defaults
        try:
            logging.info("Restoring thermal settings to safe defaults")
            
            # Gradual restoration to prevent thermal shock
            current_fan = 255 if self.current_phase >= ThermalPhase.AGGRESSIVE else 180
            target_fan = 128
            
            # Gradually reduce fan speed
            for fan_speed in range(current_fan, target_fan, -10):
                self._apply_fan_control(fan_speed)
                time.sleep(0.1)
            
            self._apply_fan_control(target_fan)
            self._apply_cpu_control(100)
            self._apply_turbo_control(True)
            
            time.sleep(1.0)
            logging.info("Thermal settings restored successfully")
            
        except Exception as e:
            logging.error(f"Failed to restore thermal settings: {e}")
        
        # Release lock
        self._release_lock()
        
        # Final diagnostics
        try:
            diag = self.run_diagnostics()
            logging.info(f"Final diagnostics: {json.dumps(diag, indent=2)}")
        except:
            pass
        
        logging.info("Thermal Guardian v3.0 shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Enhanced signal handling"""
        signal_names = {
            signal.SIGTERM: 'SIGTERM',
            signal.SIGINT: 'SIGINT',
            signal.SIGHUP: 'SIGHUP'
        }
        
        signal_name = signal_names.get(signum, f'Signal {signum}')
        logging.info(f"Received {signal_name} - initiating graceful shutdown")
        
        if signum == signal.SIGHUP:
            # Reload configuration
            try:
                self.config = self._load_config(self.config.get('config_path'))
                logging.info("Configuration reloaded")
                return
            except Exception as e:
                logging.error(f"Failed to reload config: {e}")
        
        self.running = False
        
        # Handle repeated SIGINT
        if signum == signal.SIGINT:
            if hasattr(self, '_sigint_count'):
                self._sigint_count += 1
                if self._sigint_count >= 2:
                    logging.warning("Received SIGINT twice - forcing immediate exit")
                    self._release_lock()
                    os._exit(1)
            else:
                self._sigint_count = 1
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        current_temp, confidence, valid_sensors = self.get_composite_temperature()
        
        if self.config['enable_neural_prediction']:
            predicted_temp = self.predict_temperature_neural()
        else:
            predicted_temp = self.predict_temperature_simple()
        
        # Get current control state
        current_fan = current_cpu = turbo_state = None
        try:
            with open(self.hardware.FAN_CONTROL, 'r') as f:
                current_fan = int(f.read().strip())
            with open(self.hardware.CPU_FREQ_CONTROL, 'r') as f:
                current_cpu = int(f.read().strip())
            with open(self.hardware.TURBO_CONTROL, 'r') as f:
                turbo_state = f.read().strip() == "0"
        except:
            pass
        
        phase_info = self.phases[self.current_phase]
        
        return {
            'version': '3.0',
            'temperature': {
                'current': current_temp,
                'predicted': predicted_temp,
                'confidence': confidence,
                'trend': self._calculate_trend()
            },
            'phase': {
                'current': self.current_phase.value,
                'name': phase_info['name'],
                'icon': phase_info['icon'],
                'time_in_phase': time.time() - self.phase_entry_time
            },
            'sensors': {
                'valid': valid_sensors,
                'total': len(self.sensors),
                'details': {name: {
                    'temperature': s.last_reading.temperature if s.last_reading else None,
                    'healthy': s.is_healthy(),
                    'weight': s.weight
                } for name, s in self.sensors.items()}
            },
            'controls': {
                'fan_pwm': current_fan,
                'cpu_limit': current_cpu,
                'turbo_enabled': turbo_state
            },
            'metrics': {
                'uptime': self.metrics.uptime,
                'success_rate': 1.0 - (self.metrics.failed_readings / 
                                     max(1, self.metrics.total_readings)),
                'phase_transitions': self.metrics.phase_transitions,
                'emergency_events': self.metrics.emergency_events,
                'max_temperature': self.metrics.max_temperature,
                'avg_temperature': self.metrics.avg_temperature,
                'prediction_accuracy': self.metrics.prediction_accuracy,
                'control_efficiency': self.metrics.control_efficiency
            },
            'features': {
                'neural_prediction': self.config['enable_neural_prediction'],
                'adaptive_pid': self.config['enable_adaptive_pid'],
                'sensor_fusion': 'kalman'
            }
        }
    
    def _calculate_trend(self) -> str:
        """Calculate temperature trend"""
        if len(self.thermal_history) < 10:
            return "stable"
            
        recent = list(self.thermal_history)[-10:]
        start_temp = recent[0].composite_temp
        end_temp = recent[-1].composite_temp
        
        change = end_temp - start_temp
        if change > 2.0:
            return "rising"
        elif change < -2.0:
            return "falling"
        else:
            return "stable"

def print_status(status: Dict):
    """Pretty print status"""
    temp = status['temperature']['current']
    phase_info = status['phase']
    metrics = status['metrics']
    
    # Color based on temperature
    if temp > 95:
        color = '\033[91m'  # Red
    elif temp > 85:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[92m'  # Green
    
    print(f"\n{color}{'='*60}\033[0m")
    print(f"{color}Thermal Guardian v3.0 Status\033[0m")
    print(f"{color}{'='*60}\033[0m")
    
    print(f"\n📊 TEMPERATURE")
    print(f"  Current:    {temp:.1f}°C {phase_info['icon']}")
    print(f"  Predicted:  {status['temperature']['predicted']:.1f}°C")
    print(f"  Confidence: {status['temperature']['confidence']:.1%}")
    print(f"  Trend:      {status['temperature']['trend']}")
    
    print(f"\n🔥 THERMAL PHASE")
    print(f"  Phase:      {phase_info['name']}")
    print(f"  Duration:   {phase_info['time_in_phase']:.0f}s")
    
    print(f"\n🌡️  SENSORS")
    print(f"  Active:     {status['sensors']['valid']}/{status['sensors']['total']}")
    for name, details in status['sensors']['details'].items():
        if details['temperature']:
            health = "✓" if details['healthy'] else "✗"
            print(f"  {name:15} {details['temperature']:5.1f}°C [{health}] (weight: {details['weight']:.2f})")
    
    print(f"\n⚙️  CONTROLS")
    controls = status['controls']
    print(f"  Fan Speed:  {controls['fan_pwm'] or 'N/A'}/255")
    print(f"  CPU Limit:  {controls['cpu_limit'] or 'N/A'}%")
    print(f"  Turbo:      {'Enabled' if controls['turbo_enabled'] else 'Disabled'}")
    
    print(f"\n📈 METRICS")
    print(f"  Uptime:     {metrics['uptime']/3600:.1f} hours")
    print(f"  Success:    {metrics['success_rate']:.1%}")
    print(f"  Max Temp:   {metrics['max_temperature']:.1f}°C")
    print(f"  Avg Temp:   {metrics['avg_temperature']:.1f}°C")
    print(f"  Prediction: {metrics['prediction_accuracy']:.1%} accurate")
    print(f"  Efficiency: {metrics['control_efficiency']:.1%}")
    
    print(f"\n🚨 EVENTS")
    print(f"  Phase Changes: {metrics['phase_transitions']}")
    print(f"  Emergencies:   {metrics['emergency_events']}")
    
    print(f"\n🧠 FEATURES")
    features = status['features']
    print(f"  Neural Prediction: {'✓' if features['neural_prediction'] else '✗'}")
    print(f"  Adaptive PID:      {'✓' if features['adaptive_pid'] else '✗'}")
    print(f"  Sensor Fusion:     {features['sensor_fusion'].upper()}")
    
    print(f"\n{color}{'='*60}\033[0m")

def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(
        description="Thermal Guardian v3.0 - Enhanced Edition",
        epilog="Advanced features: Neural prediction, adaptive PID, Kalman filtering"
    )
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--status', action='store_true', help='Show detailed status')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics')
    parser.add_argument('--metrics', action='store_true', help='Show metrics')
    parser.add_argument('--version', action='version', version='Thermal Guardian 3.0')
    parser.add_argument('--test-sensors', action='store_true', help='Test sensors')
    parser.add_argument('--emergency-test', action='store_true', help='Test emergency system')
    
    args = parser.parse_args()
    
    # Test sensors
    if args.test_sensors:
        print("Testing thermal sensors...")
        hw = HardwareInterface()
        sensors = hw.get_sensor_paths()
        
        print(f"\nFound {len(sensors)} sensors:")
        for name, path in sensors.items():
            try:
                with open(path, 'r') as f:
                    temp = int(f.read()) / 1000.0
                    print(f"  {name:20} {temp:6.1f}°C  [{path}]")
            except Exception as e:
                print(f"  {name:20} ERROR: {e}")
        return 0
    
    # Check permissions
    if os.geteuid() != 0:
        print("Error: Thermal Guardian requires root privileges")
        print("Run with: sudo thermal_guardian")
        return 1
    
    # Show status
    if args.status:
        try:
            guardian = ThermalGuardianV3(args.config)
            status = guardian.get_status()
            print_status(status)
            guardian._release_lock()
        except RuntimeError as e:
            if "already running" in str(e):
                print("Note: Thermal Guardian is running, showing last saved state")
                try:
                    with open(STATE_PATH, 'r') as f:
                        state = json.load(f)
                    print(f"\nLast update: {time.ctime(state['timestamp'])}")
                    print(f"Phase: {state['phase']}")
                    print(f"Max temp: {state['metrics']['max_temperature']:.1f}°C")
                except:
                    print("Unable to read saved state")
            else:
                print(f"Error: {e}")
                return 1
        return 0
    
    # Show metrics
    if args.metrics:
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            print(json.dumps(metrics, indent=2))
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return 1
        return 0
    
    # Run diagnostics
    if args.diagnostics:
        try:
            guardian = ThermalGuardianV3(args.config)
            diag = guardian.run_diagnostics()
            print(json.dumps(diag, indent=2))
            guardian._release_lock()
        except Exception as e:
            print(f"Diagnostics failed: {e}")
            return 1
        return 0
    
    # Emergency test
    if args.emergency_test:
        print("Testing emergency system (5 second test)...")
        try:
            guardian = ThermalGuardianV3(args.config)
            print("Activating emergency cooling...")
            guardian._apply_fan_control(255)
            guardian._apply_cpu_control(40)
            time.sleep(5)
            print("Restoring normal operation...")
            guardian._apply_fan_control(128)
            guardian._apply_cpu_control(100)
            guardian._release_lock()
            print("Emergency test complete")
        except Exception as e:
            print(f"Emergency test failed: {e}")
            return 1
        return 0
    
    # Start thermal guardian
    try:
        print("Starting Thermal Guardian v3.0...")
        guardian = ThermalGuardianV3(args.config)
        
        if args.daemon:
            # Simple daemonization
            if os.fork() > 0:
                return 0
            os.setsid()
            os.chdir('/')
            
        guardian.start()
        
    except RuntimeError as e:
        if "already running" in str(e):
            print("Error: Another instance of Thermal Guardian is already running")
            print("Stop existing instance with: sudo systemctl stop thermal-guardian")
        else:
            print(f"Fatal error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Unhandled exception in main")
        return 1

if __name__ == '__main__':
    sys.exit(main())

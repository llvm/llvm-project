#!/usr/bin/env python3
"""
DSMIL Structured Logging Library

Provides comprehensive logging with structured output formats for automation,
audit trails, and real-time monitoring.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class LogFormat(Enum):
    """Log output formats"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"

class DSMILLogger:
    """Structured logger for DSMIL operations"""

    def __init__(self, log_dir: str = "logs", log_format: LogFormat = LogFormat.TEXT,
                 min_level: LogLevel = LogLevel.INFO):
        self.log_dir = log_dir
        self.log_format = log_format
        self.min_level = min_level

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"dsmil_{timestamp}.log")

        # Statistics
        self.log_count = 0
        self.error_count = 0
        self.warning_count = 0

    def log(self, level: LogLevel, category: str, message: str,
            device_id: Optional[int] = None, data: Optional[Dict] = None):
        """
        Write structured log entry

        Args:
            level: Log severity level
            category: Log category (e.g., "probe", "scan", "safety")
            message: Log message
            device_id: Optional device ID
            data: Optional additional data dictionary
        """
        if level.value < self.min_level.value:
            return  # Skip logs below minimum level

        timestamp = datetime.now().isoformat()
        self.log_count += 1

        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self.error_count += 1
        elif level == LogLevel.WARNING:
            self.warning_count += 1

        # Build log entry
        entry = {
            "timestamp": timestamp,
            "level": level.name,
            "category": category,
            "message": message,
            "log_id": self.log_count,
        }

        if device_id is not None:
            entry["device_id"] = f"0x{device_id:04X}"

        if data:
            entry["data"] = data

        # Output to file and console
        self._write_to_file(entry)
        self._write_to_console(entry, level)

    def debug(self, category: str, message: str, **kwargs):
        """Log debug message"""
        self.log(LogLevel.DEBUG, category, message, **kwargs)

    def info(self, category: str, message: str, **kwargs):
        """Log info message"""
        self.log(LogLevel.INFO, category, message, **kwargs)

    def warning(self, category: str, message: str, **kwargs):
        """Log warning message"""
        self.log(LogLevel.WARNING, category, message, **kwargs)

    def error(self, category: str, message: str, **kwargs):
        """Log error message"""
        self.log(LogLevel.ERROR, category, message, **kwargs)

    def critical(self, category: str, message: str, **kwargs):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, category, message, **kwargs)

    def log_operation(self, operation: str, device_id: int, success: bool,
                     duration: float = None, result: Any = None):
        """Log a device operation"""
        level = LogLevel.INFO if success else LogLevel.ERROR

        data = {
            "operation": operation,
            "success": success,
        }

        if duration is not None:
            data["duration_ms"] = round(duration * 1000, 2)

        if result is not None:
            data["result"] = result

        message = f"Operation '{operation}' {'succeeded' if success else 'failed'}"
        self.log(level, "operation", message, device_id=device_id, data=data)

    def log_device_probe(self, device_id: int, phase: str, result: Dict):
        """Log device probing result"""
        self.info("probe", f"Phase {phase} complete",
                 device_id=device_id, data=result)

    def log_safety_check(self, device_id: int, check_name: str,
                        passed: bool, details: str = None):
        """Log safety check result"""
        level = LogLevel.INFO if passed else LogLevel.WARNING

        data = {
            "check": check_name,
            "passed": passed,
        }

        if details:
            data["details"] = details

        message = f"Safety check '{check_name}' {'passed' if passed else 'failed'}"
        self.log(level, "safety", message, device_id=device_id, data=data)

    def log_system_health(self, metrics: Dict):
        """Log system health metrics"""
        self.info("monitor", "System health check", data=metrics)

    def _write_to_file(self, entry: Dict):
        """Write log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                if self.log_format == LogFormat.JSON:
                    f.write(json.dumps(entry) + '\n')
                elif self.log_format == LogFormat.CSV:
                    # CSV format: timestamp, level, category, device_id, message
                    device_id = entry.get('device_id', 'N/A')
                    line = f"{entry['timestamp']},{entry['level']},{entry['category']},{device_id},{entry['message']}\n"
                    f.write(line)
                else:  # TEXT
                    device_str = f" [Device {entry['device_id']}]" if 'device_id' in entry else ""
                    line = f"[{entry['timestamp']}] {entry['level']:8} {entry['category']:12}{device_str} {entry['message']}\n"
                    if 'data' in entry:
                        line += f"  Data: {json.dumps(entry['data'], indent=2)}\n"
                    f.write(line)

        except Exception as e:
            print(f"Error writing to log file: {e}", file=sys.stderr)

    def _write_to_console(self, entry: Dict, level: LogLevel):
        """Write log entry to console with color"""
        # ANSI color codes
        colors = {
            LogLevel.DEBUG: '\033[90m',     # Gray
            LogLevel.INFO: '\033[0m',       # Default
            LogLevel.WARNING: '\033[93m',   # Yellow
            LogLevel.ERROR: '\033[91m',     # Red
            LogLevel.CRITICAL: '\033[1;91m', # Bold Red
        }
        reset = '\033[0m'

        color = colors.get(level, '')

        device_str = f" [0x{entry.get('device_id', 'N/A')}]" if 'device_id' in entry else ""
        timestamp = entry['timestamp'].split('T')[1][:8]  # HH:MM:SS

        print(f"{color}[{timestamp}] {entry['level']:8} {entry['category']:12}{device_str} {entry['message']}{reset}")

        # Print data if present and not debug level
        if 'data' in entry and level.value >= LogLevel.INFO.value:
            print(f"  {json.dumps(entry['data'], indent=2)}")

    def get_statistics(self) -> Dict:
        """Get logging statistics"""
        return {
            "total_logs": self.log_count,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "log_file": self.log_file,
        }

    def close(self):
        """Close logger and write summary"""
        stats = self.get_statistics()
        self.info("logger", "Logging session complete", data=stats)

def create_logger(log_dir: str = "logs",
                 log_format: LogFormat = LogFormat.TEXT,
                 min_level: LogLevel = LogLevel.INFO) -> DSMILLogger:
    """Factory function to create a logger"""
    return DSMILLogger(log_dir, log_format, min_level)

if __name__ == "__main__":
    # Self-test
    print("DSMIL Structured Logging - Self Test")
    print("=" * 80)

    # Create test logger
    logger = create_logger(log_dir="output/test_logs", min_level=LogLevel.DEBUG)

    # Test different log levels
    logger.debug("test", "This is a debug message")
    logger.info("test", "This is an info message")
    logger.warning("test", "This is a warning message")
    logger.error("test", "This is an error message")

    # Test device-specific logging
    logger.info("test", "Device-specific message", device_id=0x8003)

    # Test operation logging
    logger.log_operation("read_register", 0x8005, success=True,
                        duration=0.001234, result={"value": 0xDEADBEEF})

    # Test probe logging
    logger.log_device_probe(0x8030, "reconnaissance",
                           {"capabilities": 0x0F, "version": 1})

    # Test safety logging
    logger.log_safety_check(0x8009, "quarantine_check",
                           passed=False, details="Device is quarantined")

    # Test system health logging
    logger.log_system_health({
        "uptime": 123456,
        "load_average": 2.5,
        "memory_available_mb": 2048,
    })

    # Show statistics
    print("\n" + "=" * 80)
    print("Logger Statistics:")
    stats = logger.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    logger.close()

    print(f"\nLog file created at: {logger.log_file}")

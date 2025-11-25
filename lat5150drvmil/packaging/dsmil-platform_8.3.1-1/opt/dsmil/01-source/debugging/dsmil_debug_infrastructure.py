#!/usr/bin/env python3
"""
DSMIL Debug Infrastructure - Comprehensive Debugging and Analysis

Implements deep debugging capabilities for DSMIL SMBIOS token responses.
Provides kernel message tracing, state tracking, and correlation analysis.

Dell Latitude 5450 MIL-SPEC - 72 Device DSMIL System
Target: 0x0480-0x04C7 tokens (72 total, 6 groups of 12)
"""

import os
import sys
import time
import json
import subprocess
import threading
import queue
import re
import signal
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
import struct
import mmap
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/dsmil_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TokenState:
    """SMBIOS token state information"""
    token_id: int
    group_id: int
    device_id: int
    current_value: Optional[int] = None
    previous_value: Optional[int] = None
    access_count: int = 0
    last_access: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass
class KernelMessage:
    """Kernel message with timestamp and parsing"""
    timestamp: datetime
    facility: str
    level: str
    message: str
    raw_line: str
    is_dsmil: bool = False
    token_id: Optional[int] = None
    operation: Optional[str] = None

@dataclass
class SystemCall:
    """System call trace information"""
    timestamp: datetime
    pid: int
    process: str
    syscall: str
    args: List[str]
    return_value: Optional[int] = None
    duration: Optional[float] = None

@dataclass
class MemoryAccess:
    """Memory access pattern"""
    timestamp: datetime
    address: int
    size: int
    operation: str  # read/write
    value: Optional[int] = None
    context: str = ""

@dataclass
class CorrelationEvent:
    """Event correlation information"""
    timestamp: datetime
    event_type: str  # token_change, kernel_msg, syscall, memory_access
    details: Dict[str, Any]
    related_events: List[str] = None

class DSMILDebugger:
    """Main DSMIL debugging infrastructure"""
    
    def __init__(self, debug_dir: str = "/tmp/dsmil_debug"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        # Token tracking
        self.tokens: Dict[int, TokenState] = {}
        self.token_range = range(0x0480, 0x04C8)  # 72 tokens
        
        # Event queues
        self.kernel_messages = deque(maxlen=10000)
        self.syscalls = deque(maxlen=5000)
        self.memory_accesses = deque(maxlen=5000)
        self.correlations = deque(maxlen=5000)
        
        # Pattern recognition
        self.patterns = {
            'token_access': re.compile(r'token.*0x([0-9a-fA-F]+)'),
            'dsmil_signature': re.compile(r'DSMIL|dsmil|SMIL|DSML'),
            'group_activation': re.compile(r'Group (\d+).*activ'),
            'device_state': re.compile(r'Device.*(\d+)\.(\d+).*state.*(\w+)'),
            'memory_mapping': re.compile(r'mapping.*0x([0-9a-fA-F]+)'),
            'acpi_method': re.compile(r'ACPI.*method.*(\w+)'),
        }
        
        # Monitoring threads
        self.monitors = {}
        self.running = False
        
        # State tracking
        self.baseline_state = {}
        self.current_state = {}
        
        # Initialize token states
        self._initialize_token_states()
        
        logger.info(f"DSMIL Debugger initialized, monitoring {len(self.token_range)} tokens")

    def _initialize_token_states(self):
        """Initialize token state tracking"""
        for token_id in self.token_range:
            group_id = (token_id - 0x0480) // 12
            device_id = (token_id - 0x0480) % 12
            self.tokens[token_id] = TokenState(
                token_id=token_id,
                group_id=group_id,
                device_id=device_id
            )

    def start_monitoring(self):
        """Start all monitoring threads"""
        self.running = True
        
        # Start kernel message monitor
        self.monitors['kernel'] = threading.Thread(
            target=self._monitor_kernel_messages,
            daemon=True
        )
        self.monitors['kernel'].start()
        
        # Start system call tracer
        self.monitors['syscalls'] = threading.Thread(
            target=self._monitor_syscalls,
            daemon=True
        )
        self.monitors['syscalls'].start()
        
        # Start memory access monitor
        self.monitors['memory'] = threading.Thread(
            target=self._monitor_memory_access,
            daemon=True
        )
        self.monitors['memory'].start()
        
        # Start correlation engine
        self.monitors['correlation'] = threading.Thread(
            target=self._correlation_engine,
            daemon=True
        )
        self.monitors['correlation'].start()
        
        logger.info("All monitoring threads started")

    def stop_monitoring(self):
        """Stop all monitoring threads"""
        self.running = False
        for thread in self.monitors.values():
            thread.join(timeout=5)
        logger.info("All monitoring threads stopped")

    def _monitor_kernel_messages(self):
        """Monitor kernel messages for DSMIL activity"""
        try:
            proc = subprocess.Popen(
                ['journalctl', '-f', '-k', '--no-pager'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            logger.info("Kernel message monitoring started")
            
            while self.running and proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    continue
                    
                msg = self._parse_kernel_message(line.strip())
                if msg:
                    self.kernel_messages.append(msg)
                    
                    if msg.is_dsmil:
                        logger.info(f"DSMIL kernel message: {msg.message}")
                        self._handle_dsmil_kernel_message(msg)
                        
        except Exception as e:
            logger.error(f"Kernel message monitoring error: {e}")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()

    def _parse_kernel_message(self, line: str) -> Optional[KernelMessage]:
        """Parse kernel message line"""
        try:
            # Parse journalctl output format
            parts = line.split(' ', 3)
            if len(parts) < 4:
                return None
                
            timestamp_str = f"{parts[0]} {parts[1]}"
            timestamp = datetime.strptime(timestamp_str, "%b %d %H:%M:%S")
            timestamp = timestamp.replace(year=datetime.now().year)
            
            facility = parts[2] if ':' not in parts[2] else parts[2].split(':')[0]
            message = parts[3] if len(parts) > 3 else ""
            
            # Check if DSMIL-related
            is_dsmil = bool(self.patterns['dsmil_signature'].search(message))
            
            # Extract token ID if present
            token_match = self.patterns['token_access'].search(message)
            token_id = None
            if token_match:
                try:
                    token_id = int(token_match.group(1), 16)
                    if token_id in self.token_range:
                        is_dsmil = True
                except ValueError:
                    pass
            
            return KernelMessage(
                timestamp=timestamp,
                facility=facility,
                level="info",  # Default level
                message=message,
                raw_line=line,
                is_dsmil=is_dsmil,
                token_id=token_id
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse kernel message: {line[:100]}... Error: {e}")
            return None

    def _handle_dsmil_kernel_message(self, msg: KernelMessage):
        """Handle DSMIL-specific kernel messages"""
        # Update token state if token ID identified
        if msg.token_id and msg.token_id in self.tokens:
            token = self.tokens[msg.token_id]
            token.access_count += 1
            token.last_access = msg.timestamp
            
            # Log token activity
            logger.info(f"Token 0x{token.token_id:04X} accessed (Group {token.group_id}, Device {token.device_id})")
        
        # Check for group activation patterns
        group_match = self.patterns['group_activation'].search(msg.message)
        if group_match:
            group_id = int(group_match.group(1))
            logger.info(f"Group {group_id} activation detected")
        
        # Check for device state changes
        device_match = self.patterns['device_state'].search(msg.message)
        if device_match:
            group_id, device_id, state = device_match.groups()
            logger.info(f"Device {group_id}.{device_id} state changed to {state}")

    def _monitor_syscalls(self):
        """Monitor system calls related to SMBIOS/DSMIL"""
        try:
            # Use strace to monitor specific processes
            proc = subprocess.Popen([
                'strace', '-f', '-e', 'openat,read,write,ioctl,mmap',
                '-p', '1'  # Monitor init and children
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info("System call monitoring started")
            
            while self.running and proc.poll() is None:
                line = proc.stderr.readline()
                if not line:
                    continue
                    
                syscall = self._parse_syscall(line.strip())
                if syscall:
                    self.syscalls.append(syscall)
                    
        except subprocess.CalledProcessError:
            logger.warning("System call monitoring requires root privileges")
        except Exception as e:
            logger.error(f"System call monitoring error: {e}")

    def _parse_syscall(self, line: str) -> Optional[SystemCall]:
        """Parse strace output line"""
        try:
            # Simple strace parsing - would need more robust implementation
            if not ('(' in line and ')' in line):
                return None
                
            # Extract basic information
            parts = line.split('(', 1)
            if len(parts) < 2:
                return None
                
            syscall_name = parts[0].split()[-1] if parts[0] else "unknown"
            
            return SystemCall(
                timestamp=datetime.now(),
                pid=os.getpid(),  # Placeholder
                process="unknown",
                syscall=syscall_name,
                args=[]
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse syscall: {line[:50]}... Error: {e}")
            return None

    def _monitor_memory_access(self):
        """Monitor memory access patterns"""
        try:
            # Monitor /proc/iomem for changes
            last_iomem = self._read_iomem()
            
            while self.running:
                time.sleep(1)
                current_iomem = self._read_iomem()
                
                # Check for changes
                changes = self._compare_iomem(last_iomem, current_iomem)
                for change in changes:
                    self.memory_accesses.append(change)
                    logger.debug(f"Memory change: {change}")
                
                last_iomem = current_iomem
                
        except Exception as e:
            logger.error(f"Memory access monitoring error: {e}")

    def _read_iomem(self) -> Dict[str, str]:
        """Read /proc/iomem for memory mapping information"""
        try:
            with open('/proc/iomem', 'r') as f:
                return {line.split(':')[0].strip(): line.split(':', 1)[1].strip() 
                       for line in f if ':' in line}
        except Exception:
            return {}

    def _compare_iomem(self, old: Dict, new: Dict) -> List[MemoryAccess]:
        """Compare iomem snapshots for changes"""
        changes = []
        
        for addr, desc in new.items():
            if addr not in old or old[addr] != desc:
                try:
                    addr_int = int(addr.split('-')[0], 16)
                    changes.append(MemoryAccess(
                        timestamp=datetime.now(),
                        address=addr_int,
                        size=0,  # Unknown size
                        operation="mapping_change",
                        context=desc
                    ))
                except ValueError:
                    continue
                    
        return changes

    def _correlation_engine(self):
        """Correlate events across different monitoring streams"""
        correlation_window = 5.0  # seconds
        
        while self.running:
            try:
                # Correlate recent events within time window
                now = datetime.now()
                cutoff = now.timestamp() - correlation_window
                
                recent_kernel = [msg for msg in self.kernel_messages 
                               if msg.timestamp.timestamp() > cutoff]
                recent_syscalls = [sc for sc in self.syscalls 
                                 if sc.timestamp.timestamp() > cutoff]
                recent_memory = [ma for ma in self.memory_accesses 
                               if ma.timestamp.timestamp() > cutoff]
                
                # Look for correlations
                correlations = self._find_correlations(recent_kernel, recent_syscalls, recent_memory)
                for corr in correlations:
                    self.correlations.append(corr)
                    logger.info(f"Correlation found: {corr.event_type}")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Correlation engine error: {e}")

    def _find_correlations(self, kernel_msgs, syscalls, memory_accesses) -> List[CorrelationEvent]:
        """Find correlations between different event types"""
        correlations = []
        
        # Token access followed by kernel message
        for msg in kernel_msgs:
            if msg.token_id:
                related_events = []
                
                # Find related syscalls within 1 second
                for sc in syscalls:
                    if abs((msg.timestamp - sc.timestamp).total_seconds()) < 1.0:
                        related_events.append(f"syscall:{sc.syscall}")
                
                # Find related memory accesses
                for ma in memory_accesses:
                    if abs((msg.timestamp - ma.timestamp).total_seconds()) < 1.0:
                        related_events.append(f"memory:{ma.operation}")
                
                if related_events:
                    correlations.append(CorrelationEvent(
                        timestamp=msg.timestamp,
                        event_type="token_correlation",
                        details={
                            "token_id": msg.token_id,
                            "message": msg.message
                        },
                        related_events=related_events
                    ))
        
        return correlations

    def test_token_response(self, token_id: int, operation: str = "read") -> Dict[str, Any]:
        """Test specific token and analyze response"""
        if token_id not in self.tokens:
            return {"error": f"Token 0x{token_id:04X} not in monitored range"}
        
        logger.info(f"Testing token 0x{token_id:04X} ({operation})")
        
        # Record baseline state
        baseline_time = datetime.now()
        baseline_kernel_count = len(self.kernel_messages)
        baseline_syscall_count = len(self.syscalls)
        baseline_memory_count = len(self.memory_accesses)
        
        # Perform token operation (placeholder - would need actual SMBIOS call)
        # This would be replaced with actual dell-smbios token access
        test_result = self._simulate_token_access(token_id, operation)
        
        # Wait for system response
        time.sleep(2)
        
        # Analyze response
        response_time = datetime.now()
        new_kernel_count = len(self.kernel_messages)
        new_syscall_count = len(self.syscalls)
        new_memory_count = len(self.memory_accesses)
        
        # Collect events that occurred during test
        test_events = {
            "kernel_messages": list(self.kernel_messages)[baseline_kernel_count:],
            "syscalls": list(self.syscalls)[baseline_syscall_count:],
            "memory_accesses": list(self.memory_accesses)[baseline_memory_count:],
            "correlations": [c for c in self.correlations 
                           if baseline_time <= c.timestamp <= response_time]
        }
        
        # Update token state
        token = self.tokens[token_id]
        token.access_count += 1
        token.last_access = baseline_time
        
        if test_result.get("error"):
            token.error_count += 1
            token.last_error = test_result["error"]
        
        result = {
            "token_id": f"0x{token_id:04X}",
            "group_id": token.group_id,
            "device_id": token.device_id,
            "operation": operation,
            "timestamp": baseline_time.isoformat(),
            "duration": (response_time - baseline_time).total_seconds(),
            "test_result": test_result,
            "events_generated": {
                "kernel_messages": new_kernel_count - baseline_kernel_count,
                "syscalls": new_syscall_count - baseline_syscall_count,
                "memory_accesses": new_memory_count - baseline_memory_count
            },
            "events": {k: [asdict(e) if hasattr(e, '__dict__') else str(e) for e in v] 
                      for k, v in test_events.items()}
        }
        
        # Save detailed results
        result_file = self.debug_dir / f"token_test_{token_id:04X}_{int(baseline_time.timestamp())}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Token test complete: {result['events_generated']}")
        return result

    def _simulate_token_access(self, token_id: int, operation: str) -> Dict[str, Any]:
        """Simulate token access (placeholder for actual implementation)"""
        # This would be replaced with actual dell-smbios token access
        return {
            "simulated": True,
            "token_id": token_id,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }

    def generate_debug_report(self) -> str:
        """Generate comprehensive debug report"""
        report_file = self.debug_dir / f"debug_report_{int(datetime.now().timestamp())}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_stats": {
                "kernel_messages": len(self.kernel_messages),
                "syscalls": len(self.syscalls),
                "memory_accesses": len(self.memory_accesses),
                "correlations": len(self.correlations)
            },
            "token_states": {f"0x{tid:04X}": asdict(state) 
                           for tid, state in self.tokens.items()},
            "recent_events": {
                "kernel_messages": [asdict(msg) for msg in list(self.kernel_messages)[-10:]],
                "correlations": [asdict(corr) for corr in list(self.correlations)[-10:]]
            },
            "patterns_detected": self._analyze_patterns(),
            "system_state": self._capture_system_state()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Debug report saved: {report_file}")
        return str(report_file)

    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze detected patterns"""
        patterns = {
            "token_access_frequency": {},
            "group_activity": defaultdict(int),
            "error_patterns": [],
            "temporal_patterns": []
        }
        
        # Analyze token access patterns
        for token_id, token in self.tokens.items():
            if token.access_count > 0:
                patterns["token_access_frequency"][f"0x{token_id:04X}"] = token.access_count
                patterns["group_activity"][token.group_id] += token.access_count
        
        # Analyze error patterns
        for msg in self.kernel_messages:
            if "error" in msg.message.lower() or "fail" in msg.message.lower():
                patterns["error_patterns"].append({
                    "timestamp": msg.timestamp.isoformat(),
                    "message": msg.message
                })
        
        return patterns

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "temperature": self._get_temperature(),
                "kernel_version": subprocess.getoutput("uname -r"),
                "uptime": subprocess.getoutput("uptime"),
                "dsmil_module_loaded": "dsmil_72dev" in subprocess.getoutput("lsmod")
            }
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {"error": str(e)}

    def _get_temperature(self) -> Optional[float]:
        """Get system temperature"""
        try:
            temp_files = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
            if temp_files:
                with open(temp_files[0]) as f:
                    return float(f.read().strip()) / 1000.0
        except Exception:
            pass
        return None


class DSMILDebugCLI:
    """Command-line interface for DSMIL debugging"""
    
    def __init__(self):
        self.debugger = DSMILDebugger()
        self.running = False

    def start_interactive_session(self):
        """Start interactive debugging session"""
        print("DSMIL Debug Infrastructure - Interactive Session")
        print("Dell Latitude 5450 MIL-SPEC - 72 Device Analysis")
        print("="*60)
        
        self.debugger.start_monitoring()
        self.running = True
        
        try:
            while self.running:
                self._show_menu()
                choice = input("\nEnter choice: ").strip()
                
                if choice == '1':
                    self._monitor_realtime()
                elif choice == '2':
                    self._test_token_range()
                elif choice == '3':
                    self._analyze_correlations()
                elif choice == '4':
                    self._generate_report()
                elif choice == '5':
                    self._show_system_status()
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.debugger.stop_monitoring()
            print("Debug session ended")

    def _show_menu(self):
        """Show interactive menu"""
        print("\n" + "="*60)
        print("DSMIL DEBUG MENU")
        print("1. Real-time monitoring")
        print("2. Test token range")
        print("3. Analyze correlations")
        print("4. Generate debug report")
        print("5. Show system status")
        print("q. Quit")

    def _monitor_realtime(self):
        """Real-time monitoring display"""
        print("Real-time monitoring (Ctrl+C to stop)")
        try:
            while True:
                stats = {
                    "Kernel messages": len(self.debugger.kernel_messages),
                    "System calls": len(self.debugger.syscalls),
                    "Memory accesses": len(self.debugger.memory_accesses),
                    "Correlations": len(self.debugger.correlations)
                }
                
                print(f"\r{stats}", end='', flush=True)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

    def _test_token_range(self):
        """Test specific token range"""
        start = input("Start token (hex, e.g., 0480): ")
        end = input("End token (hex, e.g., 048F): ")
        
        try:
            start_int = int(start, 16)
            end_int = int(end, 16)
            
            print(f"Testing tokens 0x{start_int:04X} to 0x{end_int:04X}")
            
            for token_id in range(start_int, end_int + 1):
                if token_id in self.debugger.tokens:
                    result = self.debugger.test_token_response(token_id)
                    print(f"Token 0x{token_id:04X}: {result['events_generated']}")
                else:
                    print(f"Token 0x{token_id:04X}: Not monitored")
                    
        except ValueError:
            print("Invalid hex values")

    def _analyze_correlations(self):
        """Analyze event correlations"""
        print("Recent correlations:")
        for corr in list(self.debugger.correlations)[-10:]:
            print(f"{corr.timestamp}: {corr.event_type} - {len(corr.related_events or [])} related")

    def _generate_report(self):
        """Generate debug report"""
        report_file = self.debugger.generate_debug_report()
        print(f"Debug report generated: {report_file}")

    def _show_system_status(self):
        """Show current system status"""
        state = self.debugger._capture_system_state()
        for key, value in state.items():
            print(f"{key}: {value}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSMIL Debug Infrastructure")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive session")
    parser.add_argument("--test-token", "-t", type=str,
                       help="Test specific token (hex)")
    parser.add_argument("--monitor-time", "-m", type=int, default=60,
                       help="Monitor for specified seconds")
    parser.add_argument("--debug-dir", "-d", type=str, default="/tmp/dsmil_debug",
                       help="Debug output directory")
    
    args = parser.parse_args()
    
    if args.interactive:
        cli = DSMILDebugCLI()
        cli.start_interactive_session()
    elif args.test_token:
        debugger = DSMILDebugger(args.debug_dir)
        debugger.start_monitoring()
        try:
            token_id = int(args.test_token, 16)
            result = debugger.test_token_response(token_id)
            print(json.dumps(result, indent=2, default=str))
        finally:
            debugger.stop_monitoring()
    else:
        # Default monitoring
        debugger = DSMILDebugger(args.debug_dir)
        debugger.start_monitoring()
        try:
            print(f"Monitoring for {args.monitor_time} seconds...")
            time.sleep(args.monitor_time)
            report = debugger.generate_debug_report()
            print(f"Report generated: {report}")
        finally:
            debugger.stop_monitoring()


if __name__ == "__main__":
    main()
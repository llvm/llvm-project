#!/usr/bin/env python3
"""
Memory Pattern Analyzer for DSMIL Systems

Analyzes memory access patterns, MMIO operations, and memory mapping changes
related to DSMIL device operations. Provides deep analysis of memory behavior.

Dell Latitude 5450 MIL-SPEC - Memory Region 0x52000000-0x68800000 (360MB)
"""

import os
import sys
import time
import mmap
import struct
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple, Any
import logging
import subprocess
import json
import numpy as np
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryAccess:
    """Memory access record"""
    timestamp: datetime
    address: int
    size: int
    operation: str  # read, write, map, unmap
    value: Optional[int] = None
    old_value: Optional[int] = None
    context: str = ""
    caller: str = ""

@dataclass
class MemoryMapping:
    """Memory mapping record"""
    timestamp: datetime
    virtual_addr: int
    physical_addr: int
    size: int
    flags: str
    mapping_type: str  # ioremap, mmap, etc.
    process: str = ""

@dataclass
class MemoryPattern:
    """Detected memory pattern"""
    timestamp: datetime
    pattern_type: str
    description: str
    addresses: List[int]
    frequency: float
    confidence: float
    metadata: Dict[str, Any] = None

class MemoryPatternAnalyzer:
    """Advanced memory pattern analysis for DSMIL systems"""
    
    def __init__(self, analysis_dir: str = "/tmp/dsmil_memory"):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # DSMIL memory constants
        self.dsmil_base = 0x52000000
        self.dsmil_size = 360 * 1024 * 1024  # 360MB
        self.dsmil_end = self.dsmil_base + self.dsmil_size
        self.chunk_size = 4 * 1024 * 1024  # 4MB chunks
        
        # Memory tracking
        self.memory_accesses = deque(maxlen=100000)
        self.memory_mappings = deque(maxlen=10000)
        self.detected_patterns = deque(maxlen=1000)
        
        # Pattern detection parameters
        self.pattern_window = 60.0  # seconds
        self.hotspot_threshold = 10  # accesses per minute
        self.sequence_threshold = 3  # minimum sequence length
        
        # Memory state tracking
        self.memory_map = {}  # address -> last access info
        self.access_frequency = defaultdict(int)
        self.access_patterns = defaultdict(list)
        
        # Monitoring state
        self.monitoring = False
        self.threads = {}
        
        # Initialize memory monitoring
        self._init_memory_monitoring()
        
        logger.info("Memory Pattern Analyzer initialized")

    def _init_memory_monitoring(self):
        """Initialize memory monitoring capabilities"""
        # Check if we can access memory debugging features
        self.can_access_proc_mem = os.path.exists('/proc/self/mem')
        self.can_access_iomem = os.path.exists('/proc/iomem')
        self.can_use_perf = subprocess.run(['which', 'perf'], capture_output=True).returncode == 0
        
        # Log available capabilities
        capabilities = []
        if self.can_access_proc_mem:
            capabilities.append("proc_mem")
        if self.can_access_iomem:
            capabilities.append("iomem")
        if self.can_use_perf:
            capabilities.append("perf")
        
        logger.info(f"Memory monitoring capabilities: {', '.join(capabilities) or 'limited'}")

    def start_monitoring(self):
        """Start memory pattern monitoring"""
        self.monitoring = True
        
        # Start I/O memory monitoring
        if self.can_access_iomem:
            self.threads['iomem'] = threading.Thread(
                target=self._monitor_iomem_changes,
                daemon=True
            )
            self.threads['iomem'].start()
        
        # Start kernel memory events monitoring
        self.threads['kernel_mem'] = threading.Thread(
            target=self._monitor_kernel_memory_events,
            daemon=True
        )
        self.threads['kernel_mem'].start()
        
        # Start pattern detection
        self.threads['pattern_detection'] = threading.Thread(
            target=self._pattern_detection_loop,
            daemon=True
        )
        self.threads['pattern_detection'].start()
        
        # Start memory access simulation (for testing)
        if '--simulate' in sys.argv:
            self.threads['simulation'] = threading.Thread(
                target=self._simulate_memory_accesses,
                daemon=True
            )
            self.threads['simulation'].start()
        
        logger.info("Memory pattern monitoring started")

    def stop_monitoring(self):
        """Stop memory pattern monitoring"""
        self.monitoring = False
        for thread in self.threads.values():
            thread.join(timeout=5)
        self._save_analysis_results()
        logger.info("Memory pattern monitoring stopped")

    def record_memory_access(self, address: int, size: int, operation: str,
                            value: Optional[int] = None, old_value: Optional[int] = None,
                            context: str = "", caller: str = ""):
        """Record a memory access"""
        access = MemoryAccess(
            timestamp=datetime.now(),
            address=address,
            size=size,
            operation=operation,
            value=value,
            old_value=old_value,
            context=context,
            caller=caller
        )
        
        self.memory_accesses.append(access)
        self._update_access_tracking(access)
        
        # Check if this is in DSMIL region
        if self.dsmil_base <= address < self.dsmil_end:
            logger.debug(f"DSMIL memory access: 0x{address:08X} {operation} (size {size})")
            self._analyze_dsmil_access(access)

    def record_memory_mapping(self, virtual_addr: int, physical_addr: int,
                             size: int, flags: str, mapping_type: str,
                             process: str = ""):
        """Record a memory mapping operation"""
        mapping = MemoryMapping(
            timestamp=datetime.now(),
            virtual_addr=virtual_addr,
            physical_addr=physical_addr,
            size=size,
            flags=flags,
            mapping_type=mapping_type,
            process=process
        )
        
        self.memory_mappings.append(mapping)
        
        # Check if this maps DSMIL region
        if (physical_addr >= self.dsmil_base and 
            physical_addr < self.dsmil_end):
            logger.info(f"DSMIL memory mapping: 0x{physical_addr:08X}->0x{virtual_addr:08X} "
                       f"({size} bytes, {mapping_type})")

    def _monitor_iomem_changes(self):
        """Monitor /proc/iomem for memory mapping changes"""
        last_iomem = self._read_iomem()
        
        while self.monitoring:
            try:
                time.sleep(2)  # Check every 2 seconds
                current_iomem = self._read_iomem()
                
                # Detect changes
                changes = self._compare_iomem(last_iomem, current_iomem)
                for change in changes:
                    logger.debug(f"I/O memory change: {change}")
                    # Record as memory mapping if in DSMIL region
                    if self._is_dsmil_region(change.get('address', 0)):
                        self.record_memory_mapping(
                            virtual_addr=0,  # Unknown virtual address
                            physical_addr=change.get('address', 0),
                            size=change.get('size', 0),
                            flags=change.get('flags', ''),
                            mapping_type='iomem_change',
                            process='kernel'
                        )
                
                last_iomem = current_iomem
                
            except Exception as e:
                logger.error(f"I/O memory monitoring error: {e}")
                time.sleep(5)

    def _read_iomem(self) -> Dict[str, Dict]:
        """Read and parse /proc/iomem"""
        iomem_data = {}
        
        try:
            with open('/proc/iomem', 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        addr_part, desc = line.split(':', 1)
                        addr_part = addr_part.strip()
                        desc = desc.strip()
                        
                        # Parse address range
                        if '-' in addr_part:
                            start_str, end_str = addr_part.split('-')
                            try:
                                start_addr = int(start_str, 16)
                                end_addr = int(end_str, 16)
                                size = end_addr - start_addr + 1
                                
                                iomem_data[addr_part] = {
                                    'start': start_addr,
                                    'end': end_addr,
                                    'size': size,
                                    'description': desc
                                }
                            except ValueError:
                                continue
        except Exception as e:
            logger.debug(f"Failed to read /proc/iomem: {e}")
        
        return iomem_data

    def _compare_iomem(self, old: Dict, new: Dict) -> List[Dict]:
        """Compare two iomem snapshots for changes"""
        changes = []
        
        # Check for new entries
        for addr_range, info in new.items():
            if addr_range not in old:
                changes.append({
                    'type': 'new_mapping',
                    'address': info['start'],
                    'size': info['size'],
                    'description': info['description']
                })
        
        # Check for removed entries
        for addr_range, info in old.items():
            if addr_range not in new:
                changes.append({
                    'type': 'removed_mapping',
                    'address': info['start'],
                    'size': info['size'],
                    'description': info['description']
                })
        
        return changes

    def _is_dsmil_region(self, address: int) -> bool:
        """Check if address is in DSMIL memory region"""
        return self.dsmil_base <= address < self.dsmil_end

    def _monitor_kernel_memory_events(self):
        """Monitor kernel messages for memory-related events"""
        try:
            proc = subprocess.Popen(
                ['journalctl', '-f', '-k', '--no-pager'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            while self.monitoring and proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    continue
                
                self._process_kernel_memory_message(line.strip())
                
        except Exception as e:
            logger.error(f"Kernel memory event monitoring error: {e}")

    def _process_kernel_memory_message(self, message: str):
        """Process kernel message for memory-related events"""
        # Look for memory mapping messages
        import re
        
        patterns = [
            (r'ioremap.*0x([0-9a-fA-F]+).*size.*0x([0-9a-fA-F]+)', 'ioremap'),
            (r'iounmap.*0x([0-9a-fA-F]+)', 'iounmap'),
            (r'mapping.*0x([0-9a-fA-F]+)', 'mapping'),
            (r'chunk.*(\d+).*0x([0-9a-fA-F]+)', 'chunk_mapping'),
            (r'dsmil.*memory.*0x([0-9a-fA-F]+)', 'dsmil_memory')
        ]
        
        for pattern, event_type in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    if event_type == 'ioremap':
                        addr = int(match.group(1), 16)
                        size = int(match.group(2), 16)
                        if self._is_dsmil_region(addr):
                            self.record_memory_mapping(
                                virtual_addr=0,  # Will be filled by kernel
                                physical_addr=addr,
                                size=size,
                                flags='ioremap',
                                mapping_type='ioremap',
                                process='kernel'
                            )
                    elif event_type == 'chunk_mapping':
                        chunk_id = int(match.group(1))
                        addr = int(match.group(2), 16)
                        if self._is_dsmil_region(addr):
                            self.record_memory_mapping(
                                virtual_addr=0,
                                physical_addr=addr,
                                size=self.chunk_size,
                                flags=f'chunk_{chunk_id}',
                                mapping_type='chunk_mapping',
                                process='dsmil_driver'
                            )
                    elif event_type == 'dsmil_memory':
                        addr = int(match.group(1), 16)
                        if self._is_dsmil_region(addr):
                            self.record_memory_access(
                                address=addr,
                                size=4,  # Assume 32-bit access
                                operation='kernel_access',
                                context=message[:100],
                                caller='dsmil_driver'
                            )
                except ValueError:
                    continue

    def _update_access_tracking(self, access: MemoryAccess):
        """Update memory access tracking structures"""
        # Update access frequency
        self.access_frequency[access.address] += 1
        
        # Update access patterns
        self.access_patterns[access.address].append({
            'timestamp': access.timestamp,
            'operation': access.operation,
            'size': access.size
        })
        
        # Keep only recent patterns (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        for addr in self.access_patterns:
            self.access_patterns[addr] = [
                p for p in self.access_patterns[addr]
                if p['timestamp'] > cutoff
            ]
        
        # Update memory map
        self.memory_map[access.address] = {
            'last_access': access.timestamp,
            'last_operation': access.operation,
            'access_count': self.access_frequency[access.address],
            'last_value': access.value
        }

    def _analyze_dsmil_access(self, access: MemoryAccess):
        """Analyze DSMIL-specific memory access"""
        # Calculate which chunk this access falls into
        chunk_id = (access.address - self.dsmil_base) // self.chunk_size
        chunk_offset = (access.address - self.dsmil_base) % self.chunk_size
        
        # Calculate group and device if this looks like device register access
        group_stride = 0x10000  # 64KB per group
        device_stride = 0x1000   # 4KB per device
        
        if chunk_offset < (6 * group_stride):  # Within the 6 groups
            group_id = chunk_offset // group_stride
            device_offset = chunk_offset % group_stride
            if device_offset < (12 * device_stride):  # Within 12 devices
                device_id = device_offset // device_stride
                register_offset = device_offset % device_stride
                
                logger.debug(f"DSMIL device access: Group {group_id}, Device {device_id}, "
                           f"Register 0x{register_offset:03X}")
                
                # Store structured access info
                access.context = f"Group{group_id}_Device{device_id}_Reg0x{register_offset:03X}"

    def _pattern_detection_loop(self):
        """Main pattern detection loop"""
        while self.monitoring:
            try:
                self._detect_access_patterns()
                self._detect_hotspots()
                self._detect_sequences()
                self._detect_anomalies()
                time.sleep(5)  # Run detection every 5 seconds
            except Exception as e:
                logger.error(f"Pattern detection error: {e}")

    def _detect_access_patterns(self):
        """Detect memory access patterns"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.pattern_window)
        
        # Get recent accesses
        recent_accesses = [
            access for access in self.memory_accesses
            if access.timestamp > window_start
        ]
        
        if len(recent_accesses) < 5:
            return
        
        # Group by address
        addr_groups = defaultdict(list)
        for access in recent_accesses:
            addr_groups[access.address].append(access)
        
        # Look for patterns
        for addr, accesses in addr_groups.items():
            if len(accesses) >= 3:
                self._analyze_address_pattern(addr, accesses)

    def _analyze_address_pattern(self, address: int, accesses: List[MemoryAccess]):
        """Analyze access pattern for a specific address"""
        if not accesses:
            return
        
        # Calculate timing intervals
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i].timestamp - accesses[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return
        
        # Check for regular intervals (periodic access)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals) if len(intervals) > 1 else 0
        
        if std_interval < avg_interval * 0.2:  # Low variance = regular pattern
            pattern = MemoryPattern(
                timestamp=datetime.now(),
                pattern_type='periodic_access',
                description=f'Regular access pattern at 0x{address:08X}, interval ~{avg_interval:.2f}s',
                addresses=[address],
                frequency=1.0 / avg_interval if avg_interval > 0 else 0,
                confidence=0.9 - (std_interval / max(avg_interval, 0.1)),
                metadata={
                    'average_interval': avg_interval,
                    'std_interval': std_interval,
                    'access_count': len(accesses)
                }
            )
            
            self.detected_patterns.append(pattern)
            logger.info(f"Detected periodic access pattern: {pattern.description}")
        
        # Check for burst patterns
        burst_threshold = 1.0  # seconds
        bursts = []
        current_burst = []
        
        for access in accesses:
            if current_burst and (access.timestamp - current_burst[-1].timestamp).total_seconds() > burst_threshold:
                if len(current_burst) >= 3:
                    bursts.append(current_burst)
                current_burst = [access]
            else:
                current_burst.append(access)
        
        if len(current_burst) >= 3:
            bursts.append(current_burst)
        
        if len(bursts) >= 2:
            pattern = MemoryPattern(
                timestamp=datetime.now(),
                pattern_type='burst_access',
                description=f'Burst access pattern at 0x{address:08X}, {len(bursts)} bursts',
                addresses=[address],
                frequency=len(bursts) / self.pattern_window,
                confidence=0.8,
                metadata={
                    'burst_count': len(bursts),
                    'avg_burst_size': np.mean([len(burst) for burst in bursts])
                }
            )
            
            self.detected_patterns.append(pattern)
            logger.info(f"Detected burst access pattern: {pattern.description}")

    def _detect_hotspots(self):
        """Detect memory access hotspots"""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Count accesses per address in last minute
        recent_counts = defaultdict(int)
        for access in self.memory_accesses:
            if access.timestamp > window_start:
                recent_counts[access.address] += 1
        
        # Find hotspots
        hotspots = [
            (addr, count) for addr, count in recent_counts.items()
            if count >= self.hotspot_threshold
        ]
        
        for addr, count in hotspots:
            pattern = MemoryPattern(
                timestamp=datetime.now(),
                pattern_type='access_hotspot',
                description=f'High-frequency access at 0x{addr:08X}: {count} accesses/minute',
                addresses=[addr],
                frequency=count / 60.0,  # per second
                confidence=0.9 if count > self.hotspot_threshold * 2 else 0.7,
                metadata={'access_count': count, 'time_window': 60}
            )
            
            self.detected_patterns.append(pattern)
            logger.info(f"Detected access hotspot: {pattern.description}")

    def _detect_sequences(self):
        """Detect sequential memory access patterns"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.pattern_window)
        
        recent_accesses = [
            access for access in self.memory_accesses
            if access.timestamp > window_start
        ]
        
        if len(recent_accesses) < self.sequence_threshold:
            return
        
        # Sort by timestamp
        sorted_accesses = sorted(recent_accesses, key=lambda x: x.timestamp)
        
        # Look for sequential address patterns
        sequences = []
        current_seq = [sorted_accesses[0]]
        
        for i in range(1, len(sorted_accesses)):
            prev_access = sorted_accesses[i-1]
            curr_access = sorted_accesses[i]
            
            # Check if addresses are sequential and timing is close
            addr_diff = curr_access.address - prev_access.address
            time_diff = (curr_access.timestamp - prev_access.timestamp).total_seconds()
            
            if (abs(addr_diff) <= 16 and  # Within 16 bytes
                time_diff <= 2.0):        # Within 2 seconds
                current_seq.append(curr_access)
            else:
                if len(current_seq) >= self.sequence_threshold:
                    sequences.append(current_seq)
                current_seq = [curr_access]
        
        # Check final sequence
        if len(current_seq) >= self.sequence_threshold:
            sequences.append(current_seq)
        
        # Analyze sequences
        for seq in sequences:
            start_addr = seq[0].address
            end_addr = seq[-1].address
            duration = (seq[-1].timestamp - seq[0].timestamp).total_seconds()
            
            pattern = MemoryPattern(
                timestamp=datetime.now(),
                pattern_type='sequential_access',
                description=f'Sequential access: 0x{start_addr:08X} to 0x{end_addr:08X} ({len(seq)} accesses)',
                addresses=[access.address for access in seq],
                frequency=len(seq) / duration if duration > 0 else 0,
                confidence=0.8 if abs(end_addr - start_addr) < 1024 else 0.6,
                metadata={
                    'sequence_length': len(seq),
                    'address_range': abs(end_addr - start_addr),
                    'duration': duration
                }
            )
            
            self.detected_patterns.append(pattern)
            logger.info(f"Detected sequential access pattern: {pattern.description}")

    def _detect_anomalies(self):
        """Detect anomalous memory access patterns"""
        now = datetime.now()
        
        # Check for unusual access sizes
        recent_accesses = [
            access for access in self.memory_accesses
            if (now - access.timestamp).total_seconds() < 60
        ]
        
        if len(recent_accesses) < 10:
            return
        
        sizes = [access.size for access in recent_accesses]
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Find size anomalies
        for access in recent_accesses:
            if std_size > 0 and abs(access.size - avg_size) > 2 * std_size:
                pattern = MemoryPattern(
                    timestamp=datetime.now(),
                    pattern_type='size_anomaly',
                    description=f'Unusual access size at 0x{access.address:08X}: {access.size} bytes (avg: {avg_size:.1f})',
                    addresses=[access.address],
                    frequency=0,  # One-off event
                    confidence=0.7,
                    metadata={
                        'anomalous_size': access.size,
                        'average_size': avg_size,
                        'std_deviation': std_size
                    }
                )
                
                self.detected_patterns.append(pattern)
                logger.warning(f"Detected size anomaly: {pattern.description}")

    def _simulate_memory_accesses(self):
        """Simulate memory accesses for testing"""
        import random
        
        logger.info("Starting memory access simulation")
        
        while self.monitoring:
            try:
                # Generate random DSMIL region access
                base_offset = random.randint(0, 6 * 0x10000)  # Random group
                device_offset = random.randint(0, 12 * 0x1000)  # Random device
                register_offset = random.randint(0, 0x100) * 4  # Random register
                
                address = self.dsmil_base + base_offset + device_offset + register_offset
                size = random.choice([4, 8, 16])  # Common access sizes
                operation = random.choice(['read', 'write'])
                value = random.randint(0, 0xFFFFFFFF) if operation == 'write' else None
                
                self.record_memory_access(
                    address=address,
                    size=size,
                    operation=operation,
                    value=value,
                    context="simulation",
                    caller="simulator"
                )
                
                # Vary sleep time to create different patterns
                if random.random() < 0.3:  # 30% chance of burst
                    time.sleep(random.uniform(0.1, 0.3))
                else:
                    time.sleep(random.uniform(1.0, 3.0))
                    
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                time.sleep(1)

    def _save_analysis_results(self):
        """Save analysis results to files"""
        timestamp = int(datetime.now().timestamp())
        
        # Save memory accesses
        accesses_file = self.analysis_dir / f"memory_accesses_{timestamp}.json"
        with open(accesses_file, 'w') as f:
            json.dump([asdict(access) for access in self.memory_accesses], f, indent=2, default=str)
        
        # Save memory mappings
        mappings_file = self.analysis_dir / f"memory_mappings_{timestamp}.json"
        with open(mappings_file, 'w') as f:
            json.dump([asdict(mapping) for mapping in self.memory_mappings], f, indent=2, default=str)
        
        # Save detected patterns
        patterns_file = self.analysis_dir / f"memory_patterns_{timestamp}.json"
        with open(patterns_file, 'w') as f:
            json.dump([asdict(pattern) for pattern in self.detected_patterns], f, indent=2, default=str)
        
        logger.info(f"Analysis results saved: {accesses_file.parent}")

    def generate_memory_report(self) -> str:
        """Generate comprehensive memory analysis report"""
        report_file = self.analysis_dir / f"memory_report_{int(datetime.now().timestamp())}.json"
        
        # Analyze memory statistics
        total_accesses = len(self.memory_accesses)
        dsmil_accesses = sum(
            1 for access in self.memory_accesses
            if self._is_dsmil_region(access.address)
        )
        
        # Pattern statistics
        pattern_types = defaultdict(int)
        for pattern in self.detected_patterns:
            pattern_types[pattern.pattern_type] += 1
        
        # Access frequency analysis
        top_addresses = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Memory mapping analysis
        dsmil_mappings = [
            mapping for mapping in self.memory_mappings
            if self._is_dsmil_region(mapping.physical_addr)
        ]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_accesses": total_accesses,
                "dsmil_accesses": dsmil_accesses,
                "dsmil_percentage": (dsmil_accesses / max(1, total_accesses)) * 100,
                "memory_mappings": len(self.memory_mappings),
                "dsmil_mappings": len(dsmil_mappings),
                "detected_patterns": len(self.detected_patterns),
                "monitoring_capabilities": {
                    "proc_mem": self.can_access_proc_mem,
                    "iomem": self.can_access_iomem,
                    "perf": self.can_use_perf
                }
            },
            "pattern_statistics": dict(pattern_types),
            "top_accessed_addresses": [
                {"address": f"0x{addr:08X}", "count": count}
                for addr, count in top_addresses
            ],
            "memory_mappings": [asdict(mapping) for mapping in dsmil_mappings],
            "recent_patterns": [asdict(pattern) for pattern in list(self.detected_patterns)[-20:]],
            "memory_regions": {
                "dsmil_base": f"0x{self.dsmil_base:08X}",
                "dsmil_size": f"{self.dsmil_size // (1024*1024)}MB",
                "chunk_size": f"{self.chunk_size // (1024*1024)}MB"
            },
            "recommendations": self._generate_memory_recommendations()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Memory analysis report generated: {report_file}")
        return str(report_file)

    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory analysis recommendations"""
        recommendations = []
        
        if not self.memory_accesses:
            recommendations.append("No memory accesses recorded - verify monitoring is working")
            return recommendations
        
        dsmil_accesses = sum(
            1 for access in self.memory_accesses
            if self._is_dsmil_region(access.address)
        )
        
        if dsmil_accesses == 0:
            recommendations.append("No DSMIL region accesses detected - check driver functionality")
        
        # Check for excessive access frequency
        high_freq_addresses = [
            addr for addr, count in self.access_frequency.items()
            if count > 100  # More than 100 accesses
        ]
        
        if high_freq_addresses:
            recommendations.append(f"High frequency access detected at {len(high_freq_addresses)} addresses - monitor for performance impact")
        
        # Check pattern diversity
        pattern_types = set(pattern.pattern_type for pattern in self.detected_patterns)
        if len(pattern_types) > 5:
            recommendations.append("Multiple access patterns detected - system showing complex behavior")
        
        # Check for anomalies
        anomalies = [p for p in self.detected_patterns if 'anomaly' in p.pattern_type]
        if anomalies:
            recommendations.append(f"Memory anomalies detected ({len(anomalies)}) - investigate potential issues")
        
        return recommendations


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSMIL Memory Pattern Analyzer")
    parser.add_argument("--monitor", "-m", type=int, default=300,
                       help="Monitor for N seconds")
    parser.add_argument("--simulate", action="store_true",
                       help="Simulate memory accesses for testing")
    parser.add_argument("--output-dir", "-o", type=str, default="/tmp/dsmil_memory",
                       help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = MemoryPatternAnalyzer(args.output_dir)
    
    try:
        analyzer.start_monitoring()
        
        print(f"Monitoring memory patterns for {args.monitor} seconds...")
        if args.simulate:
            print("Simulation mode enabled")
        
        time.sleep(args.monitor)
        
        # Generate final report
        report_file = analyzer.generate_memory_report()
        print(f"Analysis complete. Report: {report_file}")
        
    except KeyboardInterrupt:
        print("\nMemory analysis interrupted")
    finally:
        analyzer.stop_monitoring()


if __name__ == "__main__":
    main()
# DSMIL Debugging Infrastructure

Comprehensive debugging and analysis tools for the Dell Latitude 5450 MIL-SPEC DSMIL system with 72 devices organized in 6 groups.

## Overview

This debugging infrastructure provides deep analysis capabilities for DSMIL SMBIOS token responses, kernel behavior, memory patterns, and system correlations. The tools are specifically designed for the Dell Latitude 5450 MIL-SPEC system running Debian Trixie with the dsmil-72dev kernel module.

## Components

### 1. DSMIL Debug Infrastructure (`dsmil_debug_infrastructure.py`)
- **Purpose**: Core debugging framework with event tracking and correlation
- **Features**:
  - Token state tracking for 0x0480-0x04C7 range (72 tokens)
  - Kernel message monitoring with DSMIL-specific pattern recognition
  - System call tracing for SMBIOS operations
  - Memory access pattern analysis
  - Real-time event correlation engine
  - Interactive debugging CLI

### 2. Kernel Trace Analyzer (`kernel_trace_analyzer.py`)
- **Purpose**: Advanced kernel message analysis for DSMIL operations
- **Features**:
  - Real-time kernel message streaming via journalctl
  - Pattern matching for DSMIL, ACPI, thermal, and error events
  - Baseline establishment for anomaly detection
  - Token access sequence analysis
  - Group activation pattern detection
  - Comprehensive trace reporting

### 3. SMBIOS Correlation Engine (`smbios_correlation_engine.py`)
- **Purpose**: Correlates SMBIOS token operations with system responses
- **Features**:
  - SQLite database for persistent correlation data
  - Multi-threaded event monitoring and correlation
  - Pattern detection algorithms (sequences, thermal, errors, activation)
  - Statistical analysis of token response timing
  - Cross-event correlation with configurable time windows
  - Machine learning-ready data structures

### 4. Memory Pattern Analyzer (`memory_pattern_analyzer.py`)
- **Purpose**: Memory access pattern analysis for DSMIL operations
- **Features**:
  - DSMIL memory region monitoring (0x52000000-0x68800000)
  - Memory mapping change detection via /proc/iomem
  - Access pattern recognition (periodic, burst, sequential)
  - Memory hotspot detection
  - Anomaly detection for unusual access patterns
  - Device register access correlation

### 5. Unified Debug Orchestrator (`unified_debug_orchestrator.py`)
- **Purpose**: Master orchestrator coordinating all debugging components
- **Features**:
  - Unified session management for all debugging tools
  - Cross-component data synchronization
  - Coordinated token testing sequences
  - Real-time alerting system
  - Comprehensive unified reporting
  - Interactive and automated operation modes

## Quick Start

### Prerequisites
```bash
# Ensure kernel module is loaded
sudo modprobe dsmil-72dev

# Install Python dependencies
pip3 install numpy psutil

# Ensure proper permissions for system monitoring
# Some features require root privileges
```

### Basic Usage

#### 1. Quick Debug Session (5 minutes)
```bash
cd /home/john/LAT5150DRVMIL/01-source/debugging
python3 unified_debug_orchestrator.py --duration 300
```

#### 2. Interactive Debugging
```bash
python3 unified_debug_orchestrator.py --interactive
```

#### 3. Token Range Testing
```bash
python3 unified_debug_orchestrator.py --test-tokens 0x0480:0x048F --duration 120
```

#### 4. Individual Component Usage
```bash
# Infrastructure debugger only
python3 dsmil_debug_infrastructure.py --interactive

# Kernel trace analysis
python3 kernel_trace_analyzer.py --trace 300 --baseline 60

# Memory pattern analysis
python3 memory_pattern_analyzer.py --monitor 300 --simulate

# Correlation analysis
python3 smbios_correlation_engine.py --monitor 300
```

## Architecture

### Token Organization
- **Total Tokens**: 72 (0x0480-0x04C7)
- **Group Structure**: 6 groups × 12 devices
- **Group IDs**: 0-5
- **Device IDs**: 0-11 within each group

### Memory Layout
- **DSMIL Base**: 0x52000000
- **Total Size**: 360MB
- **Chunk Size**: 4MB (for efficient mapping)
- **Group Stride**: 64KB (0x10000)
- **Device Stride**: 4KB (0x1000)

### Monitoring Capabilities
- **Kernel Messages**: journalctl integration with pattern matching
- **System Calls**: strace integration (requires root)
- **Memory Access**: /proc/iomem monitoring and pattern analysis
- **ACPI Events**: ACPI method call tracking
- **Thermal Events**: Temperature correlation with device operations

## Output Structure

All debugging output is organized under `/tmp/dsmil_unified_debug/` (configurable):

```
/tmp/dsmil_unified_debug/
├── infrastructure/          # Core debugging data
│   ├── token_test_*.json   # Individual token test results
│   └── debug_report_*.json # Infrastructure reports
├── kernel_trace/           # Kernel analysis data
│   ├── trace_report_*.json # Trace analysis reports
│   └── baseline_*.json     # Baseline patterns
├── correlation/            # Correlation analysis
│   ├── correlation_report_*.json
│   └── correlation.db      # SQLite database
├── memory_analysis/        # Memory pattern data
│   ├── memory_report_*.json
│   ├── memory_accesses_*.json
│   └── memory_patterns_*.json
├── intermediate_report_*.json  # Periodic session reports
└── unified_debug_report_*.json # Final comprehensive reports
```

## Advanced Usage

### Custom Configuration
```python
# Configure unified orchestrator
orchestrator = UnifiedDebugOrchestrator("/custom/debug/path")
orchestrator.configure_session(
    duration_seconds=600,
    auto_report_interval=30,
    enable_realtime_alerts=True,
    alert_thresholds={
        'error_rate': 0.05,  # 5% error rate threshold
        'memory_anomaly_count': 3
    }
)
```

### Programmatic Integration
```python
from unified_debug_orchestrator import UnifiedDebugOrchestrator

# Initialize and start debugging
orchestrator = UnifiedDebugOrchestrator()
orchestrator.start_unified_debugging()

# Execute specific token tests
results = orchestrator.execute_token_test_sequence(
    range(0x0480, 0x0490),  # Test first 16 tokens
    operations=['read', 'query']
)

# Generate report and stop
report = orchestrator.generate_unified_report()
orchestrator.stop_unified_debugging()
```

### Pattern Recognition Customization
```python
# Add custom patterns to kernel tracer
custom_patterns = {
    'custom_dsmil_error': re.compile(r'DSMIL.*error.*code\s*(\d+)', re.IGNORECASE),
    'device_timeout': re.compile(r'device.*timeout.*group\s*(\d+)', re.IGNORECASE)
}

tracer = KernelTraceAnalyzer()
tracer.patterns.update(custom_patterns)
```

## Safety Considerations

### Thermal Monitoring
- The system monitors thermal conditions and can trigger emergency stops
- Default thermal threshold: 85°C
- Monitoring integrates with existing thermal guardian system

### Safe Operation
- All operations are read-only by default
- JRTC1 training mode enforced for safety
- Emergency stop mechanisms in all components
- Graceful shutdown on system signals (SIGINT, SIGTERM)

### Permissions
- Some features require root privileges for system call tracing
- Memory monitoring uses safe read-only /proc interfaces
- Kernel message access via journalctl (may require user in systemd-journal group)

## Troubleshooting

### Common Issues

#### 1. No DSMIL Activity Detected
```bash
# Check if kernel module is loaded
lsmod | grep dsmil

# Check module parameters
cat /sys/module/dsmil_72dev/parameters/*

# Verify token range
python3 -c "
for i in range(0x480, 0x4C8):
    print(f'Token 0x{i:04X} -> Group {(i-0x480)//12}, Device {(i-0x480)%12}')
"
```

#### 2. Permission Errors
```bash
# Add user to required groups
sudo usermod -a -G systemd-journal $USER

# For system call tracing (requires root)
sudo python3 dsmil_debug_infrastructure.py --interactive
```

#### 3. High Resource Usage
- Reduce monitoring duration
- Increase sampling intervals
- Disable simulation modes
- Use component-specific tools instead of unified orchestrator

### Debug Logging
```bash
# Enable debug logging
export PYTHONPATH=/home/john/LAT5150DRVMIL/01-source/debugging
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python3 unified_debug_orchestrator.py --duration 60
```

## Integration with Existing Tools

### TESTBED Framework Integration
```python
# Use with existing testing framework
from testing.smbios_testbed_framework import SMBIOSTestbedFramework
from debugging.unified_debug_orchestrator import UnifiedDebugOrchestrator

testbed = SMBIOSTestbedFramework()
orchestrator = UnifiedDebugOrchestrator()

# Coordinate testing and debugging
orchestrator.start_unified_debugging()
testbed.run_comprehensive_tests()
debug_report = orchestrator.generate_unified_report()
orchestrator.stop_unified_debugging()
```

### Monitoring Integration
```bash
# Integrate with existing monitoring
cd /home/john/LAT5150DRVMIL/monitoring
./start_monitoring_session.sh &

# Start debugging in parallel
cd /home/john/LAT5150DRVMIL/01-source/debugging
python3 unified_debug_orchestrator.py --duration 300
```

## Report Analysis

### Understanding Reports

#### Unified Debug Report Structure
```json
{
  "metadata": {
    "timestamp": "2025-01-27T...",
    "session_duration_seconds": 300,
    "target_system": "Dell Latitude 5450 MIL-SPEC"
  },
  "component_reports": {
    "infrastructure": { /* Token tracking, correlations */ },
    "kernel_trace": { /* Kernel message analysis */ },
    "correlation": { /* Statistical correlations */ },
    "memory_analysis": { /* Memory pattern analysis */ }
  },
  "unified_findings": {
    "system_behavior": { /* Cross-component analysis */ },
    "token_patterns": { /* Token access patterns */ },
    "performance_metrics": { /* Response times, etc. */ }
  },
  "recommendations": [ /* Actionable recommendations */ ]
}
```

#### Key Metrics to Monitor
1. **Token Response Times**: Average < 50ms is normal
2. **Error Rates**: Should be < 5% for stable operation
3. **Memory Patterns**: Regular patterns indicate proper operation
4. **Correlation Strength**: High correlation (>0.8) indicates predictable behavior

## Future Enhancements

### Planned Features
- Machine learning-based anomaly detection
- Real-time dashboard web interface
- Integration with Dell service tools
- Automated root cause analysis
- Performance regression detection

### Extension Points
- Custom pattern detectors
- Additional correlation algorithms
- External data source integration
- Report format customization

## Contributing

When extending the debugging infrastructure:

1. **Follow existing patterns** for consistency
2. **Add comprehensive logging** for debugging
3. **Include error handling** for robustness
4. **Document new features** in this README
5. **Test with thermal guardian** active

## Support

For issues with the debugging infrastructure:

1. Check system logs: `journalctl -f -k | grep -i dsmil`
2. Verify kernel module status: `cat /proc/modules | grep dsmil`
3. Review thermal conditions: `/home/john/LAT5150DRVMIL/thermal_status.py`
4. Test with minimal configuration first
5. Generate debug reports for analysis

---

**Last Updated**: 2025-01-27  
**Compatibility**: Debian Trixie, Linux 6.14+, Dell Latitude 5450 MIL-SPEC  
**Module Version**: dsmil-72dev v2.0.0
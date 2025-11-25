# DSMIL Comprehensive Monitoring System

## Overview

This monitoring system provides comprehensive real-time monitoring for DSMIL SMBIOS token testing on Dell Latitude 5450 MIL-SPEC systems. It includes safety mechanisms to prevent system freeze, resource exhaustion, and thermal issues.

## Components

### Core Monitoring Scripts

#### `dsmil_comprehensive_monitor.py`
- **Purpose**: Main monitoring dashboard with multiple display modes
- **Modes**: 
  - `dashboard` - Overview of all system metrics
  - `resources` - CPU, memory, temperature focus
  - `tokens` - SMBIOS token state tracking
  - `alerts` - Real-time alert monitoring
- **Features**:
  - Real-time system metrics collection
  - Temperature monitoring with thermal safety limits
  - Memory and CPU usage tracking
  - SMBIOS token change detection
  - Kernel message monitoring
  - Emergency stop capabilities

#### `safe_token_tester.py`
- **Purpose**: Safe SMBIOS token testing with comprehensive safety checks
- **Features**:
  - Pre-test system validation
  - Resource exhaustion prevention
  - Temperature safety limits
  - Dry-run simulation mode
  - Automatic rollback on issues
  - Comprehensive result logging

#### `multi_terminal_launcher.sh`
- **Purpose**: Launch multiple monitoring terminals simultaneously
- **Terminals**:
  1. Dashboard - Main system overview
  2. Resources - System resource monitoring
  3. Tokens - SMBIOS token tracking
  4. Alerts - Real-time alert system
  5. Kernel - Kernel message monitoring

#### `emergency_stop.sh`
- **Purpose**: Immediate emergency stop for all DSMIL operations
- **Actions**:
  - Stop all monitoring processes
  - Remove DSMIL kernel modules
  - Kill SMBIOS testing scripts
  - Check thermal status
  - Create emergency log

#### `start_monitoring_session.sh`
- **Purpose**: Interactive menu system for monitoring operations
- **Features**:
  - System prerequisite checking
  - Multiple monitoring options
  - Safe token testing interface
  - Log viewing capabilities
  - Emergency controls

### Configuration Files

#### `alert_thresholds.json`
- **Purpose**: Centralized configuration for all monitoring thresholds
- **Contents**:
  - System limits (temperature, CPU, memory)
  - Monitoring settings
  - Token testing parameters
  - Emergency action procedures
  - DSMIL range definitions
  - Hardware-specific settings

## Quick Start

### 1. Interactive Menu System (Recommended)
```bash
./monitoring/start_monitoring_session.sh
```

### 2. Multi-Terminal Dashboard
```bash
./monitoring/multi_terminal_launcher.sh
```

### 3. Single Terminal Monitoring
```bash
# Dashboard view
python3 monitoring/dsmil_comprehensive_monitor.py --mode dashboard

# Resource focus
python3 monitoring/dsmil_comprehensive_monitor.py --mode resources

# Token monitoring
python3 monitoring/dsmil_comprehensive_monitor.py --mode tokens

# Alert monitoring  
python3 monitoring/dsmil_comprehensive_monitor.py --mode alerts
```

### 4. Safe Token Testing
```bash
# Dry run (simulation only)
python3 monitoring/safe_token_tester.py --range Range_0480

# Live testing (DANGEROUS - modifies SMBIOS)
python3 monitoring/safe_token_tester.py --range Range_0480 --live
```

### 5. Emergency Stop
```bash
./monitoring/emergency_stop.sh
```

## Safety Features

### Temperature Protection
- **Warning**: 85°C - Display warning alerts
- **Critical**: 90°C - Stop non-essential operations
- **Emergency**: 95°C - Full system emergency stop

### Memory Protection
- **Warning**: 80% usage - Monitor closely
- **Critical**: 90% usage - Reduce operations
- **Emergency**: 95% usage - Stop all testing

### CPU Protection
- **Warning**: 80% usage - Reduce testing frequency
- **Critical**: 90% usage - Pause testing
- **Emergency**: 95% usage - Emergency stop

### Resource Exhaustion Prevention
- Pre-test system validation
- Continuous monitoring during operations
- Automatic operation suspension on limit breach
- Emergency procedures for critical situations

## DSMIL Token Ranges

The system monitors 11 DSMIL token ranges discovered in the system:

| Range | Start | End | Priority | Description |
|-------|-------|-----|----------|-------------|
| Range_0400 | 0x0400 | 0x0447 | Low | DSMIL range 1 |
| **Range_0480** | **0x0480** | **0x04C7** | **High** | **Primary range (most promising)** |
| Range_0500 | 0x0500 | 0x0547 | Medium | DSMIL range 3 |
| Range_1000-1700 | Various | Various | Low | Additional ranges |

## Monitoring Modes

### Dashboard Mode
- Overall system status indicator
- Temperature, CPU, memory summary
- Recent alerts display
- Token change tracking
- Kernel message monitoring
- Real-time metrics updates

### Resources Mode
- Detailed CPU usage (per-core)
- Memory breakdown (used/available)
- Disk I/O statistics
- Thermal status with warnings
- Process count monitoring
- Load average tracking

### Tokens Mode
- All 11 DSMIL ranges status
- Recently changed tokens
- Active token detection
- Token state tracking
- Range-specific monitoring

### Alerts Mode
- Real-time alert stream
- Alert history (last 20 events)
- Severity level indicators
- Component-specific alerts
- Timestamp tracking

## Log Files

### Monitoring Logs
- Location: `monitoring/logs/`
- Format: `monitor_YYYYMMDD_HHMMSS.log`
- Content: All monitoring events and metrics

### Token Testing Logs
- Location: `monitoring/logs/`
- Format: `token_test_YYYYMMDD_HHMMSS.log`
- Content: Test procedures, results, safety checks

### Emergency Logs
- Location: `/tmp/dsmil_emergency_YYYYMMDD_HHMMSS.log`
- Content: Emergency stop events, system state

### Result Files
- Location: `monitoring/logs/`
- Format: `test_results_YYYYMMDD_HHMMSS.json`
- Content: Structured test results and metrics

## Integration with Existing System

### Kernel Module Integration
- Monitors `dsmil-72dev` kernel module
- Tracks module load/unload events
- Monitors for kernel message activity

### SMBIOS Integration
- Uses Dell SMBIOS interface when available
- Safe token reading with timeout protection
- Simulation mode for safe testing

### System Integration
- Works with existing thermal monitoring
- Integrates with Dell hardware modules
- Respects system resource limits

## Emergency Procedures

### Thermal Emergency (>95°C)
1. Immediate stop of all token testing
2. Kill all monitoring processes
3. Remove DSMIL kernel modules
4. Log emergency event
5. Display cooling recommendations

### Memory Emergency (>95% usage)
1. Stop resource-intensive operations
2. Kill non-essential processes
3. Force garbage collection
4. Save system state
5. Log memory statistics

### General Emergency
1. Stop all DSMIL operations
2. Create emergency snapshot
3. Save current logs
4. Display recovery instructions

## Requirements

### System Requirements
- Linux system with sysfs support
- Python 3.6+ with psutil module
- Sudo access for system monitoring
- Terminal emulator (gnome-terminal recommended)

### Hardware Requirements
- Dell Latitude 5450 MIL-SPEC (tested system)
- Thermal monitoring support
- SMBIOS/DMI interface access
- Sufficient memory for monitoring overhead

### Software Requirements
```bash
# Install Python requirements
pip3 install psutil --user

# Verify system tools
which dmesg dmidecode free top
```

## Usage Examples

### Start Complete Monitoring Session
```bash
./monitoring/start_monitoring_session.sh
# Select option 1 for multi-terminal dashboard
```

### Run Safe Token Test
```bash
# Test primary range in dry-run mode
python3 monitoring/safe_token_tester.py --range Range_0480

# View results
cat monitoring/logs/token_test_*.log
```

### Monitor During Testing
```bash
# Terminal 1: Run test
python3 monitoring/safe_token_tester.py --range Range_0480 --live

# Terminal 2: Monitor alerts
python3 monitoring/dsmil_comprehensive_monitor.py --mode alerts

# Terminal 3: Monitor resources
python3 monitoring/dsmil_comprehensive_monitor.py --mode resources
```

### Emergency Response
```bash
# If system becomes unresponsive
./monitoring/emergency_stop.sh

# Check emergency logs
ls -la /tmp/dsmil_emergency_*.log
```

## Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Fix: Ensure sudo access
echo "1786" | sudo -S echo "Test"
```

#### Python Module Missing
```bash
# Fix: Install required modules
pip3 install psutil --user
```

#### Terminal Issues
```bash
# Alternative: Use single terminal mode
python3 monitoring/dsmil_comprehensive_monitor.py --mode dashboard
```

#### High Resource Usage
```bash
# Fix: Reduce monitoring frequency
# Edit UPDATE_INTERVAL in dsmil_comprehensive_monitor.py
```

### Debug Mode

Enable detailed logging:
```bash
# Set debug environment
export DSMIL_DEBUG=1

# Run with verbose output
python3 monitoring/dsmil_comprehensive_monitor.py --mode dashboard
```

## Development Notes

### Adding New Monitoring Features
1. Update `dsmil_comprehensive_monitor.py`
2. Add new thresholds to `alert_thresholds.json`
3. Update documentation
4. Test with dry-run mode first

### Extending Token Ranges
1. Add new range to `alert_thresholds.json`
2. Update token monitor in main script
3. Test with simulation mode
4. Document range purpose

### Custom Alert Actions
1. Implement in emergency action handlers
2. Add to configuration file
3. Test emergency procedures
4. Update emergency stop script

## Security Considerations

### Sudo Usage
- Scripts use limited sudo commands
- Password stored in scripts (change as needed)
- Consider using sudo rules for specific commands

### Token Modification
- Dry-run mode prevents accidental changes
- Live mode requires explicit confirmation
- All changes are logged with timestamps
- Emergency rollback procedures available

### Log Security
- Logs may contain sensitive system information
- Restrict access to log directory
- Consider log encryption for sensitive environments

---

**Last Updated**: 2025-09-01  
**Version**: 1.0  
**Tested On**: Dell Latitude 5450 MIL-SPEC  
**Author**: MONITOR Agent
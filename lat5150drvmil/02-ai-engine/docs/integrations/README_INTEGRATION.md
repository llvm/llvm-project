# DSMIL 104-Device Integration

Complete integration of DSMIL control centres and discovery processes with the new 104-device kernel driver v5.2.0.

## Overview

This integration connects all existing DSMIL Python tools with the new kernel driver, providing:

- **Cascading auto-discovery** of all 104 devices
- **Unified control interface** for device management
- **Backward compatibility** with 84-device tools
- **Safety enforcement** with quarantine protection
- **Real-time monitoring** and diagnostics
- **TPM authentication** integration
- **Comprehensive audit trails**

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 DSMIL Control Centre                         │
│            (dsmil_control_centre_104.py)                     │
│  Interactive menu system for discovery, activation,          │
│  monitoring, and diagnostics                                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│              Integration Adapter                             │
│           (dsmil_integration_adapter.py)                     │
│  Unified API providing:                                      │
│  - Cascading device discovery (4-phase process)              │
│  - Multi-method activation (IOCTL/sysfs/SMI)                 │
│  - Legacy tool compatibility                                 │
│  - Safety enforcement and auditing                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│              Driver Interface                                │
│           (dsmil_driver_interface.py)                        │
│  Python IOCTL bindings:                                      │
│  - 12 IOCTL commands                                         │
│  - Token read/write (104 devices × 3 tokens)                 │
│  - TPM authentication                                        │
│  - BIOS management                                           │
│  - System monitoring via sysfs                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│         DSMIL Kernel Driver v5.2.0                           │
│           (dsmil-104dev.c)                                   │
│  - 104 devices (0x8000-0x8137)                               │
│  - 3 redundant BIOS systems                                  │
│  - TPM 2.0 hardware authentication                           │
│  - Real/simulated SMBIOS backend                             │
│  - Error handling and audit logging                          │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. Driver Interface (`dsmil_driver_interface.py`)

Low-level Python bindings for the kernel driver.

**Features:**
- IOCTL command wrappers for all 12 commands
- Token read/write operations
- Device discovery and enumeration
- TPM authentication interface
- BIOS failover and synchronization
- Sysfs attribute access
- Diagnostic utilities

**Usage:**
```python
from dsmil_driver_interface import DSMILDriverInterface

# Open driver
with DSMILDriverInterface() as driver:
    # Get driver version
    version = driver.get_version()
    print(f"Driver version: {version}")

    # Discover devices
    devices = driver.discover_devices()
    print(f"Found {len(devices)} devices")

    # Read token
    value = driver.read_token(0x8000)
    print(f"Token 0x8000 = 0x{value:08X}")

    # Get system status
    status = driver.get_status()
    print(f"Active BIOS: {chr(ord('A') + status.active_bios)}")
```

### 2. Extended Device Database (`dsmil_device_database_extended.py`)

Extended database supporting all 104 devices.

**Features:**
- 104 device definitions (up from 84)
- 9 device groups:
  - Groups 0-6: Original 84 devices
  - Group 7: Diagnostic Tools (12 devices, all SAFE)
  - Group 8: Advanced Features (12 devices)
  - Extended: 20 expansion slots
- Token-to-device mapping
- Safety classifications
- Backward compatibility

**Usage:**
```python
from dsmil_device_database_extended import (
    get_device_extended,
    get_token_range,
    get_statistics_extended,
    SAFE_DEVICES_EXTENDED,
    QUARANTINED_DEVICES_EXTENDED
)

# Get device info
device = get_device_extended(5)
print(f"Device 5: {device.name}")
print(f"Safe to activate: {device.safe_to_activate}")

# Get token range for device
status_token, config_token, data_token = get_token_range(5)
print(f"Device 5 tokens: 0x{status_token:04X}, 0x{config_token:04X}, 0x{data_token:04X}")

# Get statistics
stats = get_statistics_extended()
print(f"Total devices: {stats['total_devices']}")
print(f"Safe devices: {stats['safe']}")
```

### 3. Integration Adapter (`dsmil_integration_adapter.py`)

Unified adapter connecting all components.

**Features:**
- **Cascading Discovery:** 4-phase device discovery process
  1. IOCTL token scanning (primary)
  2. Sysfs enumeration (fallback)
  3. Database validation
  4. Quarantine filtering
- **Multi-Method Activation:** IOCTL, sysfs, or SMI
- **Safety Enforcement:** Quarantine blocking, safety checks
- **System Monitoring:** Thermal, BIOS, authentication status
- **Legacy Compatibility:** Works with existing 84-device tools
- **Audit Logging:** Complete activation history

**Usage:**
```python
from dsmil_integration_adapter import DSMILIntegrationAdapter

# Initialize adapter
adapter = DSMILIntegrationAdapter()

# Discover all devices
devices = adapter.discover_all_devices_cascading(
    progress_callback=lambda msg: print(msg)
)
print(f"Discovered {len(devices)} devices")

# Activate safe devices only
results = adapter.activate_safe_devices_only(
    progress_callback=lambda msg: print(msg)
)
print(f"Activated {sum(results.values())} devices")

# Get system status
status = adapter.get_system_status()
print(f"Thermal: {status.thermal_celsius}°C")
print(f"Active BIOS: {chr(ord('A') + status.active_bios)}")

# Generate report
report = adapter.generate_discovery_report()
print(f"Report: {report['devices_discovered']}/104 discovered")
```

### 4. Control Centre (`dsmil_control_centre_104.py`)

Interactive control centre for complete system management.

**Features:**
- **Discovery Mode:** Cascading scan of all 104 devices
- **Activation Mode:** Safe and custom device activation
- **Monitoring Mode:** Real-time system status display
- **Diagnostics Mode:** Comprehensive health checks
- **Reporting:** JSON export with full audit trails
- **Interactive Menu:** User-friendly interface

**Usage:**
```bash
# Interactive menu (recommended for first use)
sudo python3 dsmil_control_centre_104.py

# Auto-discovery on startup
sudo python3 dsmil_control_centre_104.py --auto-discover

# Full automation (discover + activate safe devices)
sudo python3 dsmil_control_centre_104.py \
    --auto-discover --auto-activate --non-interactive
```

## Cascading Discovery Process

The integration uses a 4-phase cascading discovery process:

### Phase 1: IOCTL Token Scanning (Primary)
- Sequentially scans all 104 devices (0-103)
- Attempts to read status token for each device
- Filters quarantined devices (0x8009, 0x800A, 0x800B, 0x8019, 0x8029)
- Returns list of responsive devices

### Phase 2: Sysfs Enumeration (Fallback)
- Falls back to sysfs if IOCTL fails
- Reads device count from `/sys/class/dsmil/dsmil0/device_count`
- Provides alternative discovery method

### Phase 3: Database Validation
- Validates discovered devices against extended database
- Retrieves device metadata (name, group, safety)
- Ensures device definitions exist

### Phase 4: Quarantine Filtering
- Final safety check against quarantine list
- Blocks dangerous devices from activation
- Logs quarantine enforcement

## Quick Start

### 1. Build and Load Driver

```bash
# Build driver
cd 01-source/kernel
make clean
make

# Load driver
sudo insmod core/dsmil-104dev.ko

# Verify driver loaded
ls -l /dev/dsmil0
dmesg | tail -20
```

### 2. Run Diagnostics

```bash
# Test driver interface
sudo python3 02-ai-engine/dsmil_driver_interface.py

# Test integration adapter
sudo python3 02-ai-engine/dsmil_integration_adapter.py
```

### 3. Launch Control Centre

```bash
# Interactive control centre
sudo python3 02-ai-engine/dsmil_control_centre_104.py
```

### 4. Discover and Activate Devices

From the control centre menu:
1. Select **Option 1: Device Discovery**
   - Scans all 104 devices
   - Shows devices by group and safety status

2. Select **Option 2: Activate Safe Devices**
   - Automatically activates verified safe devices
   - Confirmation required before activation

3. Select **Option 4: System Monitoring**
   - Real-time monitoring of system status
   - Thermal, BIOS, and device metrics

## Safety Features

### Quarantined Devices (Always Blocked)

These 5 devices are permanently quarantined and **NEVER** activated:

| Token ID | Device Name | Reason |
|----------|-------------|--------|
| 0x8009 | DATA DESTRUCTION | DOD-standard data wipe |
| 0x800A | CASCADE WIPE | Secondary wipe system |
| 0x800B | HARDWARE SANITIZE | Hardware-level sanitization |
| 0x8019 | NETWORK KILL | Network destruction |
| 0x8029 | COMMS BLACKOUT | Communications blackout |

### Safe Devices (Auto-Activate Approved)

These devices are verified safe for automatic activation:
- 0x8003-0x8007: Core monitoring devices
- 0x802A: Network monitor
- 0x8070-0x807B: All diagnostic tools (Group 7)

### Safety Checks

All activation attempts perform:
1. **Quarantine Check:** Blocks quarantined devices
2. **Database Validation:** Verifies device exists
3. **Safety Flag Check:** Confirms `safe_to_activate` flag
4. **Audit Logging:** Records all activation attempts

## Integration with Existing Tools

The integration provides backward compatibility with existing DSMIL Python tools:

### Legacy Control Centres
- `dsmil_subsystem_controller.py` - Still works via adapter
- `dsmil_operation_monitor.py` - Reads from new driver
- `dsmil_guided_activation.py` - Uses integration adapter

### Legacy Discovery Scripts
- `dsmil_discover.py` - Supplemented by cascading discovery
- `dsmil_auto_discover.py` - Works alongside new discovery
- `dsmil_ml_discovery.py` - Can use adapter for hardware access

### Legacy Activation Systems
- `dsmil_integrated_activation.py` - Compatible with adapter
- `dsmil_device_activation.py` - Uses new IOCTL interface

## Programmatic Usage Examples

### Example 1: Simple Discovery

```python
#!/usr/bin/env python3
from dsmil_integration_adapter import quick_discover

# Quick discovery (returns initialized adapter)
adapter = quick_discover()

# Print summary
adapter.print_system_summary()
```

### Example 2: Activate Specific Devices

```python
#!/usr/bin/env python3
from dsmil_integration_adapter import DSMILIntegrationAdapter

adapter = DSMILIntegrationAdapter()

# Discover devices
devices = adapter.discover_all_devices_cascading()

# Activate specific devices (example: devices 3, 4, 5)
results = adapter.activate_multiple_devices([3, 4, 5])

print(f"Results: {results}")
```

### Example 3: Monitor System

```python
#!/usr/bin/env python3
from dsmil_integration_adapter import DSMILIntegrationAdapter
import time

adapter = DSMILIntegrationAdapter()

# Monitor for 60 seconds
for i in range(12):
    status = adapter.get_system_status()

    print(f"[{i*5}s] Thermal: {status.thermal_celsius}°C, " +
          f"BIOS: {chr(ord('A') + status.active_bios)}, " +
          f"Tokens: {status.token_reads} reads")

    time.sleep(5)
```

### Example 4: Full Automation

```python
#!/usr/bin/env python3
from dsmil_integration_adapter import DSMILIntegrationAdapter

adapter = DSMILIntegrationAdapter()

# 1. Discovery
print("Phase 1: Discovery...")
devices = adapter.discover_all_devices_cascading()
print(f"  Discovered {len(devices)} devices")

# 2. Activation
print("Phase 2: Activation...")
results = adapter.activate_safe_devices_only()
print(f"  Activated {sum(results.values())} devices")

# 3. Monitoring
print("Phase 3: Monitoring...")
if adapter.check_thermal_safe():
    print("  ✓ Thermal levels safe")
else:
    print("  ⚠️  Thermal warning")

# 4. Report
print("Phase 4: Report...")
report = adapter.generate_discovery_report()
with open('/tmp/dsmil_report.json', 'w') as f:
    import json
    json.dump(report, f, indent=2)
print("  ✓ Report saved")
```

## API Reference

### DSMILDriverInterface

| Method | Description |
|--------|-------------|
| `open()` | Open driver device |
| `close()` | Close driver device |
| `get_version()` | Get driver version string |
| `get_status()` | Get system status |
| `read_token(token_id)` | Read token value |
| `write_token(token_id, value)` | Write token value |
| `get_device_info(device_id)` | Get device status/config/data |
| `discover_devices()` | Discover all devices |
| `activate_device(device_id)` | Activate single device |
| `tpm_get_challenge()` | Get TPM authentication challenge |
| `authenticate(method, data)` | Submit authentication |
| `get_bios_status()` | Get BIOS health and status |
| `bios_failover(target_bios)` | Trigger BIOS failover |

### DSMILIntegrationAdapter

| Method | Description |
|--------|-------------|
| `discover_all_devices_cascading()` | 4-phase cascading discovery |
| `activate_device(device_id)` | Activate with safety checks |
| `activate_multiple_devices(ids)` | Activate list of devices |
| `activate_safe_devices_only()` | Auto-activate safe devices |
| `get_system_status()` | Get comprehensive status |
| `get_bios_status()` | Get BIOS status |
| `get_thermal_status()` | Get thermal temperature |
| `check_thermal_safe(threshold)` | Check thermal safety |
| `get_device_status(device_id)` | Get device status |
| `is_device_activated(device_id)` | Check activation status |
| `generate_discovery_report()` | Generate JSON report |
| `print_system_summary()` | Print status summary |

## Troubleshooting

### Driver Not Found

```
✗ Driver not loaded (/dev/dsmil0 not found)
```

**Solution:**
```bash
# Load driver
sudo insmod 01-source/kernel/core/dsmil-104dev.ko

# Verify
ls -l /dev/dsmil0
```

### Permission Denied

```
PermissionError: [Errno 13] Permission denied: '/dev/dsmil0'
```

**Solution:**
```bash
# Run with sudo
sudo python3 dsmil_control_centre_104.py
```

### No Devices Discovered

```
Discovered 0/104 devices
```

**Solutions:**
1. Check driver is loaded: `lsmod | grep dsmil`
2. Check device permissions: `ls -l /dev/dsmil0`
3. Check kernel logs: `dmesg | grep DSMIL`
4. Verify SMBIOS backend: `cat /sys/class/dsmil/dsmil0/smbios_backend`

### Import Errors

```
ImportError: No module named 'dsmil_driver_interface'
```

**Solution:**
```bash
# Ensure correct path
cd LAT5150DRVMIL
export PYTHONPATH=$PWD/02-ai-engine:$PYTHONPATH

# Or run from correct directory
cd 02-ai-engine
python3 dsmil_control_centre_104.py
```

## Documentation Links

- **Driver Documentation:**
  - [DRIVER_USAGE_GUIDE.md](../01-source/kernel/DRIVER_USAGE_GUIDE.md) - Driver usage and IOCTL interface
  - [API_REFERENCE.md](../01-source/kernel/API_REFERENCE.md) - Complete API documentation
  - [TPM_AUTHENTICATION_GUIDE.md](../01-source/kernel/TPM_AUTHENTICATION_GUIDE.md) - TPM authentication guide
  - [TESTING_GUIDE.md](../01-source/kernel/TESTING_GUIDE.md) - Comprehensive testing procedures

- **Legacy Documentation:**
  - [FULL_DEVICE_COVERAGE_ANALYSIS.md](FULL_DEVICE_COVERAGE_ANALYSIS.md) - Original 84-device analysis
  - Device implementation files in `02-tools/dsmil-devices/devices/`

## Support

For issues or questions:
1. Check kernel logs: `dmesg | grep DSMIL`
2. Review driver documentation in `01-source/kernel/`
3. Run diagnostics: `python3 dsmil_control_centre_104.py` → Option 5
4. Generate report: `python3 dsmil_control_centre_104.py` → Option 6

---

**Version:** 1.0.0
**Driver Compatibility:** dsmil-104dev v5.2.0
**Last Updated:** 2025-11-13
**Status:** Production Ready

# DSMIL Integration Automation Suite

**Automated tools for safe exploration and integration of 84 DSMIL hardware devices**

## 🎯 Overview

This automation suite provides production-ready tools for safely exploring, documenting, and integrating the Dell System Management Interface Layer (DSMIL) devices on the Dell Latitude 5450 MIL-SPEC platform. The framework implements a 5-phase progressive exploration methodology with comprehensive safety checks.

### Features

- ✅ **Automated Safe Probing** - Progressive 5-phase device exploration
- ✅ **Device Capability Scanner** - Fast enumeration of all 84 devices
- ✅ **Real-Time Safety Monitoring** - Continuous system health checks
- ✅ **Auto-Documentation Generator** - Automatic markdown documentation
- ✅ **Multi-Layer Safety System** - Hardware, kernel, and software protection
- ✅ **Structured Logging** - JSON, CSV, and text formats
- ✅ **Emergency Stop** - Automatic rollback on critical conditions

## 📊 Device Coverage

| Status | Devices | Percentage | Description |
|--------|---------|------------|-------------|
| ✅ **Implemented** | 6 | 7.1% | Fully monitored and safe |
| 🔴 **Quarantined** | 5 | 6.0% | Permanently blocked (destructive) |
| ⚠️ **Identified** | 6 | 7.1% | Known but not yet integrated |
| ❓ **Unknown** | 67 | 79.8% | Requires exploration |
| **TOTAL** | **84** | **100%** | Complete device space |

## 🛠️ Tool Suite

### 1. `dsmil_probe.py` - Automated Safe Probing Framework

Progressive device exploration using 5-phase methodology.

**Usage:**
```bash
# Probe single device (reconnaissance only - safest)
sudo python3 dsmil_probe.py --device 0x8030

# Probe with full observation (Phase 2)
sudo python3 dsmil_probe.py --device 0x8030 --phase 2

# Probe entire group (Group 3: Data Processing)
sudo python3 dsmil_probe.py --range 0x8030 0x803B

# Dry run (no hardware access)
python3 dsmil_probe.py --device 0x8030 --dry-run
```

**Phases:**
1. **RECONNAISSANCE** - Passive capability reading (safest)
2. **PASSIVE OBSERVATION** - Read-only monitoring (safe)
3. **CONTROLLED TESTING** - Single operations (requires manual approval)
4. **INCREMENTAL ENABLING** - Supervised activation (manual only)
5. **PRODUCTION INTEGRATION** - Full deployment (manual only)

### 2. `dsmil_scanner.py` - Device Capability Scanner

Fast scanning of devices without deep probing.

**Usage:**
```bash
# Quick scan (no hardware access)
python3 dsmil_scanner.py --quick

# Full scan with hardware probing
sudo python3 dsmil_scanner.py

# Scan specific group
sudo python3 dsmil_scanner.py --group 3

# Export to JSON
python3 dsmil_scanner.py --quick --export devices.json
```

### 3. `dsmil_monitor.py` - Real-Time Safety Monitor

Continuous monitoring of system health and safety metrics.

**Usage:**
```bash
# Start monitoring (1 second interval)
sudo python3 dsmil_monitor.py

# Custom interval
sudo python3 dsmil_monitor.py --interval 0.5

# With verbose logging
sudo python3 dsmil_monitor.py --verbose
```

**Monitors:**
- System uptime and load average
- Available memory
- Disk space
- Thermal status
- Device states
- Automatic emergency stop on critical conditions

### 4. `dsmil_docgen.py` - Auto-Documentation Generator

Generate comprehensive markdown documentation from probe results.

**Usage:**
```bash
# Generate all documentation
python3 dsmil_docgen.py --input scan_results.json

# Generate device profile
python3 dsmil_docgen.py --input scan_results.json --device 0x8030

# Generate group summary
python3 dsmil_docgen.py --input scan_results.json --group 3

# Generate master index
python3 dsmil_docgen.py --input scan_results.json --index
```

## 📁 Directory Structure

```
dsmil-explorer/
├── lib/                          # Core libraries
│   ├── dsmil_safety.py           # Safety validation & device risk
│   ├── dsmil_common.py           # Device access & utilities
│   └── dsmil_logger.py           # Structured logging
├── dsmil_probe.py                # Automated probing framework
├── dsmil_scanner.py              # Device capability scanner
├── dsmil_monitor.py              # Real-time safety monitoring
├── dsmil_docgen.py               # Auto-documentation generator
├── config/                       # Configuration files
├── output/                       # Output directory
│   ├── probe_logs/               # Probe logs
│   ├── scan_logs/                # Scanner logs
│   ├── monitor_logs/             # Monitor logs
│   └── docs/                     # Generated documentation
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

1. **DSMIL Kernel Module Loaded:**
```bash
cd /home/user/LAT5150DRVMIL/01-source/kernel
sudo insmod dsmil-72dev.ko
```

2. **Verify Module:**
```bash
lsmod | grep dsmil
ls -l /dev/dsmil0
```

3. **Permissions:**
Run with `sudo` or add user to `dsmil` group.

### Example Workflow

**Step 1: Quick Scan (No Hardware Access)**
```bash
# Get overview of all devices
python3 dsmil_scanner.py --quick --export devices_overview.json
```

**Step 2: Start Safety Monitor (Optional)**
```bash
# In separate terminal
sudo python3 dsmil_monitor.py
```

**Step 3: Probe Unknown Devices**
```bash
# Probe Group 3 (Data Processing) - 12 devices
sudo python3 dsmil_probe.py --range 0x8030 0x803B --phase 2
```

**Step 4: Generate Documentation**
```bash
# Generate comprehensive docs
python3 dsmil_docgen.py --input output/probe_logs/latest_results.json
```

**Step 5: Review Results**
```bash
# Check generated documentation
ls output/docs/
cat output/docs/DEVICE_INDEX.md
```

## 🛡️ Safety Features

### Multi-Layer Protection

1. **Hardware Level** - SMI interface blocks dangerous operations
2. **Kernel Level** - DSMIL driver enforces quarantine
3. **Security Level** - Access control and authorization
4. **Software Level** - Safety validation and emergency stop
5. **Application Level** - Structured logging and monitoring

### Quarantined Devices (NEVER ACCESS)

```
0x8009 - DATA_DESTRUCTION      (DOD wipe)
0x800A - CASCADE_WIPE          (Secondary wipe)
0x800B - HARDWARE_SANITIZE     (Final destruction)
0x8019 - NETWORK_KILL          (Network destruction)
0x8029 - COMMS_BLACKOUT        (Communications kill)
```

These devices are **permanently blocked** at all levels.

### Emergency Stop Conditions

Automatic emergency stop triggered on:
- Load average > 20
- Available memory < 50 MB
- Disk space < 100 MB
- Thermal alert
- Unexpected device state changes

## 📊 Expected Results

### Phase 1 (Reconnaissance) - Typical Results
```
Device 0x8030 (Data Processing):
  ✓ Capabilities read
  ✓ Device status retrieved
  ✓ Version detected
  Duration: ~50ms
```

### Phase 2 (Passive Observation) - Typical Results
```
Device 0x8030 (Data Processing):
  ✓ Register values read (16 bytes)
  ✓ 3 status samples collected
  ✓ No anomalies detected
  Duration: ~3 seconds
```

## 🔧 Configuration

### Safety Profiles

Edit `lib/dsmil_safety.py` to customize:
- Device risk classifications
- Safety thresholds
- Emergency stop conditions

### Logging

Edit logging settings in each tool:
```python
logger = create_logger(
    log_dir="output/logs",
    log_format=LogFormat.JSON,  # TEXT, JSON, or CSV
    min_level=LogLevel.INFO     # DEBUG, INFO, WARNING, ERROR, CRITICAL
)
```

## 📈 Progress Tracking

### Current Status (As of Implementation)

**Explored Devices:**
- Group 0: 6/12 devices (50%) - Core security
- Group 1: 0/12 devices (0%) - Extended security
- Group 2: 1/12 devices (8%) - Network/comms
- Group 3-6: 0/48 devices (0%) - Unknown

**Next Priority:**
1. Group 3 (Data Processing) - 12 devices
2. Group 2 (Network) - Remaining 10 devices
3. Group 1 (Extended Security) - 12 devices

### Integration Timeline

| Phase | Devices | Est. Time | Status |
|-------|---------|-----------|--------|
| P1: Safe Expansion | 9 | 2-3 weeks | **Next** |
| P2: Network | 10 | 3-4 weeks | Planned |
| P3: Data Processing | 12 | 4-6 weeks | Planned |
| P4: Storage | 12 | 4-6 weeks | Planned |
| P5: Peripherals | 12 | 3-4 weeks | Planned |
| P6: Training | 12 | 3-4 weeks | Planned |

## 🐛 Troubleshooting

### Kernel Module Not Loaded
```bash
Error: DSMIL kernel module not loaded

Solution:
cd /home/user/LAT5150DRVMIL/01-source/kernel
sudo insmod dsmil-72dev.ko
dmesg | tail -20  # Check for errors
```

### Permission Denied
```bash
Error: Permission denied accessing /dev/dsmil0

Solution:
sudo python3 dsmil_probe.py ...

Or add user to group:
sudo usermod -a -G dsmil $USER
```

### Device Not Responding
```bash
Warning: Could not read device capabilities

Possible causes:
1. Device not initialized
2. Device in power-saving mode
3. Hardware protection active

Solution: Check dmesg for kernel messages
```

## 📚 Additional Documentation

- **Architecture**: See `/00-documentation/SYSTEM_ARCHITECTURE.md`
- **Device Analysis**: See `/00-documentation/02-analysis/hardware/FULL_DEVICE_COVERAGE_ANALYSIS.md`
- **Safety Guide**: See `/01-source/kernel/README_TRACK_A.md`

## 🤝 Contributing

When exploring new devices:

1. Always start with Phase 1 (Reconnaissance)
2. Log all results comprehensively
3. Generate documentation immediately
4. Update device risk classifications
5. Share findings with the team

## ⚖️ Classification

**UNCLASSIFIED // FOR OFFICIAL USE ONLY**

This automation framework is designed for safe exploration of military-grade hardware. Always follow safety protocols and never bypass quarantine protections.

## 📞 Support

For issues or questions:
1. Check logs in `output/*/`
2. Review dmesg for kernel messages
3. Consult main documentation in `/00-documentation/`

---

**Built for Dell Latitude 5450 MIL-SPEC Platform**
**DSMIL Framework - 84 Devices - Safe Exploration Automation**

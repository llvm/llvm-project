# Dell MIL-SPEC Tools - Complete File Layout

## Package File Structure

### Package Archive
```
dell-milspec-tools_1.0.0-1_amd64.deb
Size: 24,300 bytes (24 KB)
Format: Debian Binary Package (format 2.0)
Architecture: amd64
Compression: gzip
```

## Installed Files by Directory

### /usr/bin/ (User Commands)

```
/usr/bin/dsmil-status
├── Type: Bash script
├── Size: 4,195 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Query DSMIL device status
└── Dependencies: bash, lsmod, stat, cat

/usr/bin/dsmil-test
├── Type: Python 3 script
├── Size: 6,427 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Test DSMIL functionality
├── Dependencies: python3, safe_token_tester module
└── Shebang: #!/usr/bin/env python3

/usr/bin/tpm2-accel-status
├── Type: Bash script
├── Size: 4,927 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Query TPM2 acceleration status
└── Dependencies: bash, lsmod, dmesg

/usr/bin/milspec-control
├── Type: Bash script
├── Size: 7,981 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Main control utility (interactive menu)
└── Dependencies: All other dell-milspec-tools commands

/usr/bin/milspec-monitor
├── Type: Python 3 wrapper script
├── Size: 3,041 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Monitoring dashboard launcher
├── Dependencies: python3, dsmil_comprehensive_monitor module
└── Shebang: #!/usr/bin/env python3
```

**Total: 5 commands, 26,571 bytes**

### /usr/sbin/ (System Commands)

```
/usr/sbin/milspec-emergency-stop
├── Type: Bash script
├── Size: 5,017 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Emergency shutdown procedures (<85ms target)
├── Dependencies: bash, pkill, rmmod, date
└── Performance: Typically completes in 45-65ms
```

**Total: 1 command, 5,017 bytes**

### /usr/share/dell-milspec/monitoring/ (Python Modules)

```
/usr/share/dell-milspec/monitoring/dsmil_comprehensive_monitor.py
├── Type: Python 3 module
├── Size: 25,098 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Lines: 681 lines of code
├── Purpose: Real-time monitoring dashboard
├── Features:
│   ├── Multi-mode display (dashboard/alerts/resources/tokens)
│   ├── Resource tracking (thermal, CPU, memory, disk I/O)
│   ├── Token monitoring (11 ranges, 792 tokens)
│   ├── Alert system (4 levels: INFO/WARNING/CRITICAL/EMERGENCY)
│   ├── JSON output mode
│   └── Emergency stop mechanism
├── Dependencies: python3, psutil, threading, subprocess
├── Update Interval: 0.5 seconds (configurable)
└── Memory Usage: ~50 MB typical

/usr/share/dell-milspec/monitoring/safe_token_tester.py
├── Type: Python 3 module
├── Size: 15,808 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Lines: 414 lines of code
├── Purpose: Safe SMBIOS token testing
├── Features:
│   ├── Dry-run mode (default, no actual writes)
│   ├── Safety checks (thermal, memory, CPU)
│   ├── Pre/post operation validation
│   ├── Session-based logging
│   ├── Test result tracking
│   └── JSON result export
├── Dependencies: python3, subprocess, dataclasses
├── Token Ranges: 11 ranges (0x0400-0x1747)
└── Safety Checks: Every 5 seconds during testing
```

**Total: 2 modules, 40,906 bytes, 1,095 lines of code**

### /usr/share/dell-milspec/config/ (Configuration Templates)

```
/usr/share/dell-milspec/config/dsmil.conf.default
├── Type: Shell configuration file
├── Size: 4,675 bytes
├── Permissions: 0644 (rw-r--r--)
├── Format: KEY="value" shell syntax
├── Sections:
│   ├── Device Configuration
│   ├── Module Configuration
│   ├── Token Range Configuration
│   ├── Safety Limits
│   ├── Logging Configuration
│   ├── Monitoring Configuration
│   ├── Alert Configuration
│   ├── Emergency Stop Configuration
│   ├── Hardware Specific
│   ├── Integration
│   └── Advanced Options
└── Purpose: DSMIL device and module configuration

/usr/share/dell-milspec/config/monitoring.json.default
├── Type: JSON configuration file
├── Size: 4,418 bytes
├── Permissions: 0644 (rw-r--r--)
├── Format: JSON
├── Sections:
│   ├── system_limits (temperature, CPU, memory, disk I/O, load avg)
│   ├── monitoring_settings (update interval, alerts, emergency)
│   ├── token_testing (limits, delays, safety checks)
│   ├── emergency_actions (thermal, memory, general)
│   ├── alert_notifications (colors, sounds)
│   ├── dsmil_ranges (11 ranges with priorities)
│   ├── hardware_specific (Dell Latitude 5450 settings)
│   └── logging (levels, format, rotation)
└── Purpose: Monitoring thresholds and alert configuration

/usr/share/dell-milspec/config/safety.json.default
├── Type: JSON configuration file
├── Size: 4,264 bytes
├── Permissions: 0644 (rw-r--r--)
├── Format: JSON
├── Sections:
│   ├── safety_parameters (version, platform)
│   ├── thermal_safety (ranges: normal/warning/critical/emergency)
│   ├── resource_safety (CPU, memory, disk I/O, load avg limits)
│   ├── operation_limits (token testing, monitoring, emergency)
│   ├── quarantine_rules (conditions, thresholds, duration)
│   ├── pre_operation_checks (required checks, timeout)
│   ├── post_operation_validation (validations, delays)
│   ├── emergency_procedures (priorities, actions, timeouts)
│   ├── operator_notifications (events, methods)
│   ├── recovery_procedures (auto-recovery, steps)
│   └── compliance (MIL-SPEC 810H parameters)
└── Purpose: Safety parameters and emergency procedures
```

**Total: 3 configuration templates, 13,357 bytes**

### /usr/share/dell-milspec/examples/ (Usage Examples)

```
/usr/share/dell-milspec/examples/README.md
├── Type: Markdown documentation
├── Size: 2,727 bytes
├── Permissions: 0644 (rw-r--r--)
├── Purpose: Example overview and quick start guide
└── Sections: Available examples, prerequisites, usage, support

/usr/share/dell-milspec/examples/example-basic-usage.sh
├── Type: Bash script (interactive)
├── Size: 2,124 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Demonstrate basic operations
├── Steps:
│   ├── 1. Check DSMIL device status
│   ├── 2. Check TPM2 acceleration status
│   ├── 3. Run basic device tests
│   ├── 4. View configuration
│   └── 5. Start monitoring dashboard (30s demo)
└── Execution Time: ~2-3 minutes

/usr/share/dell-milspec/examples/example-monitoring.sh
├── Type: Bash script (interactive menu)
├── Size: 1,983 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Demonstrate different monitoring modes
├── Modes:
│   ├── 1. Dashboard mode
│   ├── 2. Resource mode
│   ├── 3. Token mode
│   ├── 4. Alert mode
│   └── 5. JSON output mode
└── Execution: User-driven menu system

/usr/share/dell-milspec/examples/example-token-testing.sh
├── Type: Bash script (interactive menu)
├── Size: 3,985 bytes
├── Permissions: 0755 (rwxr-xr-x)
├── Purpose: Demonstrate safe token testing
├── Options:
│   ├── 1. Test Range_0480 (dry-run)
│   ├── 2. Test Range_0400 (dry-run)
│   ├── 3. Test Range_0500 (dry-run)
│   ├── 4. Custom range testing
│   └── 5. View test logs
└── Safety: All tests in DRY RUN mode by default
```

**Total: 4 files (1 doc + 3 scripts), 10,819 bytes**

### /etc/dell-milspec/ (Configuration Files)

```
/etc/dell-milspec/
├── Type: Directory
├── Permissions: 0755 (rwxr-xr-x)
├── Created: During package installation (postinst)
├── Owner: root:root
└── Purpose: User-editable configuration files

Configuration files copied from templates during installation:

/etc/dell-milspec/dsmil.conf
├── Source: /usr/share/dell-milspec/config/dsmil.conf.default
├── Permissions: 0644 (rw-r--r--)
├── Editable: Yes (preserved on package upgrade)
└── Purpose: DSMIL device configuration

/etc/dell-milspec/monitoring.json
├── Source: /usr/share/dell-milspec/config/monitoring.json.default
├── Permissions: 0644 (rw-r--r--)
├── Editable: Yes (preserved on package upgrade)
└── Purpose: Monitoring thresholds

/etc/dell-milspec/safety.json
├── Source: /usr/share/dell-milspec/config/safety.json.default
├── Permissions: 0644 (rw-r--r--)
├── Editable: Yes (preserved on package upgrade)
└── Purpose: Safety parameters
```

**Total: 1 directory + 3 configuration files**
**Note:** Configuration files are NOT included in package archive, created by postinst

### /var/log/dell-milspec/ (Log Files)

```
/var/log/dell-milspec/
├── Type: Directory
├── Permissions: 0755 (rwxr-xr-x)
├── Created: During package installation (postinst)
├── Owner: root:root
├── Purpose: Log file storage
└── Contents (created at runtime):
    ├── system.log                    (General system logs)
    ├── token_test_YYYYMMDD_HHMMSS.log (Token test session logs)
    ├── test_results_YYYYMMDD_HHMMSS.json (Token test results)
    ├── emergency_YYYYMMDD_HHMMSS.log (Emergency stop logs)
    └── dmesg_emergency_YYYYMMDD_HHMMSS.log (Kernel messages)
```

**Total: 1 directory**
**Note:** Directory created by postinst, contents created at runtime

### /var/run/dell-milspec/ (Runtime Data)

```
/var/run/dell-milspec/
├── Type: Directory
├── Permissions: 0755 (rwxr-xr-x)
├── Created: During package installation (postinst)
├── Owner: root:root
├── Purpose: Runtime data storage (PID files, sockets)
└── Contents (created at runtime):
    ├── milspec-monitor.pid           (Monitoring daemon PID)
    └── *.tmp                         (Temporary files)
```

**Total: 1 directory**
**Note:** Directory created by postinst, contents created at runtime

### System Group

```
Group: dsmil
├── Type: System group
├── GID: Automatically assigned by system
├── Created: During package installation (postinst)
├── Purpose: Device access permissions
├── Members: Current user (added automatically)
└── Removed: During package purge (postrm)
```

**Note:** Users must logout/login after group membership change

### Device Node Permissions (Modified at Install)

```
/dev/dsmil0
├── Owner: root:dsmil (modified by postinst if device exists)
├── Permissions: 0660 (rw-rw----) (modified by postinst if device exists)
└── Purpose: DSMIL device access

/dev/tpm2_accel_early
├── Owner: root:dsmil (modified by postinst if device exists)
├── Permissions: 0660 (rw-rw----) (modified by postinst if device exists)
└── Purpose: TPM2 acceleration device access
```

**Note:** Devices must be created by kernel modules (not included in package)

## DEBIAN Control Files

Located in package metadata (not installed on system):

```
DEBIAN/control
├── Size: 1,058 bytes
└── Content: Package metadata, dependencies, description

DEBIAN/postinst
├── Size: 4,705 bytes
├── Lines: 124 lines
├── Purpose: Post-installation configuration
└── Actions:
    ├── Create directories (/etc, /var/log, /var/run)
    ├── Install default configuration files
    ├── Create dsmil group
    ├── Add current user to dsmil group
    ├── Set device permissions
    └── Display installation summary

DEBIAN/prerm
├── Size: 1,271 bytes
├── Lines: 48 lines
├── Purpose: Pre-removal cleanup
└── Actions:
    ├── Stop monitoring processes
    └── Remove PID files

DEBIAN/postrm
├── Size: 2,354 bytes
├── Lines: 78 lines
├── Purpose: Post-removal cleanup
└── Actions (on purge):
    ├── Remove configuration directory
    ├── Remove log directory
    ├── Remove runtime directory
    └── Remove dsmil group

DEBIAN/conffiles
├── Size: 0 bytes
└── Content: Empty (configs created by postinst, not in package)

DEBIAN/copyright
├── Size: 1,439 bytes
└── Content: GPL-3.0+ license information

DEBIAN/changelog
├── Size: 1,058 bytes
└── Content: Version 1.0.0-1 release notes

DEBIAN/compat
├── Size: 3 bytes
└── Content: "10" (Debhelper compatibility level)
```

## Complete File Inventory

### By Category

**Executable Commands:** 6 files, 31,588 bytes
- User commands (5): 26,571 bytes
- System commands (1): 5,017 bytes

**Python Modules:** 2 files, 40,906 bytes, 1,095 LOC

**Configuration Templates:** 3 files, 13,357 bytes

**Example Scripts:** 3 files, 8,092 bytes

**Documentation:** 1 file, 2,727 bytes

**DEBIAN Control:** 8 files, 13,848 bytes

### By Type

**Scripts (Bash):** 9 files (.sh, executables)
**Scripts (Python):** 4 files (.py, executables)
**Configuration (Shell):** 1 file (.conf)
**Configuration (JSON):** 2 files (.json)
**Documentation (Markdown):** 1 file (.md)
**Control Files:** 8 files (DEBIAN/*)

**Total Package Contents:** 23 files + 3 directories

### By Size

Largest files:
1. dsmil_comprehensive_monitor.py - 25,098 bytes
2. safe_token_tester.py - 15,808 bytes
3. milspec-control - 7,981 bytes
4. dsmil-test - 6,427 bytes
5. milspec-emergency-stop - 5,017 bytes

### By Purpose

**Status/Query:** 2 files (dsmil-status, tpm2-accel-status)
**Testing:** 1 file (dsmil-test)
**Monitoring:** 2 files (milspec-monitor, dsmil_comprehensive_monitor.py)
**Token Operations:** 1 file (safe_token_tester.py)
**Control/Management:** 1 file (milspec-control)
**Emergency:** 1 file (milspec-emergency-stop)
**Configuration:** 6 files (3 templates + 3 installed configs)
**Examples:** 4 files (1 doc + 3 scripts)
**Control/Metadata:** 8 files (DEBIAN/*)

## Installation Footprint

### Disk Space Usage

```
Package archive:        24 KB
Installed files:        ~140 KB
Configuration:          ~13 KB
Runtime logs:           Variable (depends on usage)
Total (fresh install):  ~180 KB + logs
```

### Runtime Memory Usage

```
Monitoring dashboard:   ~50 MB (Python process)
Token testing:          ~30 MB (Python process)
Bash scripts:           <5 MB each
Total (monitoring):     ~50-80 MB
```

### Network Usage

```
Package download:       24 KB
No runtime network:     0 KB (all operations local)
```

## File Permissions Summary

```
Directories:            0755 (rwxr-xr-x)
Executables:            0755 (rwxr-xr-x)
Configuration files:    0644 (rw-r--r--)
Log files:              0644 (rw-r--r--)
Device nodes:           0660 (rw-rw----)
```

## Ownership Summary

```
Package files:          root:root
Configuration:          root:root (user-editable)
Logs:                   root:root (created at runtime)
Device nodes:           root:dsmil
```

---

**Package Version:** 1.0.0-1
**Architecture:** amd64
**Format:** Debian Binary Package (format 2.0)
**Build Date:** 2025-10-11
**Compressed Size:** 24,300 bytes
**Installed Size:** ~512 KB (estimated, excluding logs)

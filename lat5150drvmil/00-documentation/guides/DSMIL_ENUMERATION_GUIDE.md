# DSMIL Device Enumeration Guide

**Complete Guide to Enumerating Dell System Military Integration Layer Devices**

**Platform:** Dell Latitude 5450 MIL-SPEC
**Coverage:** 80/108 devices (74.1%)
**Framework Version:** 2.0.0 (Auto-Discovery)
**Last Updated:** 2025-11-08

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Current Status](#current-status)
4. [Enumeration Tools](#enumeration-tools)
5. [Command-Line Interface](#command-line-interface)
6. [Device Documentation](#device-documentation)
7. [Interactive Probes](#interactive-probes)
8. [Discovery Strategy](#discovery-strategy)
9. [Safety & Security](#safety--security)
10. [Next Steps](#next-steps)

---

## Quick Start

### 1. Enumerate Devices with CLI

```bash
# Show comprehensive statistics
python3 scripts/dsmil-cli.py stats

# List all 80 implemented devices
python3 scripts/dsmil-cli.py list

# Get detailed information about a specific device
python3 scripts/dsmil-cli.py info 0x8000

# Search for devices by name
python3 scripts/dsmil-cli.py search "security"

# Show devices organized by group
python3 scripts/dsmil-cli.py groups

# Find most complex devices
python3 scripts/dsmil-cli.py complex --top 10

# Export device list to CSV
python3 scripts/dsmil-cli.py export --format csv > devices.csv
```

### 2. Run Interactive Probe (On Actual Hardware)

```bash
# Interactive DSMIL enumeration
python3 scripts/interactive-probes/04_test_dsmil_enumeration.py

# Options available:
#   1. Check DSMIL driver status
#   2. Check DSMIL device nodes
#   3. Check military token status
#   4. List implemented devices (80/108)
#   5. Launch DSMIL interactive menu (full TUI)
#   6. Run device discovery
#   7. Probe specific device
#   8. Show quick reference
#   9. Run all checks
```

### 3. Browse Device Documentation

```bash
# View index of all devices
cat 00-documentation/devices/README.md

# View specific device documentation
cat 00-documentation/devices/0x8000.md

# Search documentation
grep -r "TPM" 00-documentation/devices/
```

---

## Overview

### What is DSMIL?

**Dell System Military Integration Layer (DSMIL)** is a hardware abstraction layer providing access to 108 specialized military-grade devices on the Dell Latitude 5450 MIL-SPEC platform.

### Device Categories

| Group | Name | Range | Devices | Status |
|-------|------|-------|---------|--------|
| **0** | Core Security | 0x8000-0x800B | 12 | ✅ 100% (9 active, 3 quarantined) |
| **1** | Extended Security | 0x800C-0x8017 | 12 | ✅ 100% |
| **2** | Network/Comms | 0x8018-0x8023 | 12 | ✅ 100% (11 active, 1 quarantined) |
| **3** | Data Processing | 0x8024-0x802F | 12 | ✅ 100% (11 active, 1 quarantined) |
| **4** | Storage Management | 0x8030-0x803B | 12 | ✅ 100% |
| **5** | Peripheral Control | 0x803C-0x8047 | 12 | ✅ 100% |
| **6** | Training/Simulation | 0x8048-0x8053 | 12 | ✅ 100% |
| **Ext** | Extended Range | 0x8054-0x806B | 24 | ⚠️ 4% (1 active, 23 unknown) |
| | **TOTAL** | | **108** | **74.1%** |

### Key Features

- **656 total operations** across 80 devices
- **273 hardware registers** mapped
- **Post-Quantum Cryptography** support (FIPS 203/204)
- **Military token validation** (0x049e-0x04a3)
- **Multi-layer quarantine** for destructive devices
- **Auto-discovery system** v2.0.0

---

## Current Status

### ✅ Implemented (80 devices)

**Standard Range (Groups 0-6):** 100% complete
- All 84 standard devices fully integrated
- 5 devices quarantined for safety
- Auto-discovery and device registry operational

**Extended Range (0x8054-0x806B):** 4% complete
- 1 device integrated (0x805A: SensorArray)
- 23 devices unknown/unidentified

### 🔴 Quarantined Devices (Never Access)

| ID | Name | Purpose | Risk |
|----|------|---------|------|
| **0x8009** | DATA DESTRUCTION | DOD-level data wipe | 🔴 CRITICAL |
| **0x800A** | CASCADE WIPE | Secondary wipe system | 🔴 CRITICAL |
| **0x800B** | HARDWARE SANITIZE | Physical destruction trigger | 🔴 CRITICAL |
| **0x8019** | NETWORK KILL | Network interface destruction | 🔴 CRITICAL |
| **0x8029** | COMMS BLACKOUT | Communications kill switch | 🔴 CRITICAL |

**Protection Layers:**
- Hardware: Cannot be accessed via SMI/ACPI
- Kernel: Blocked by device drivers
- Software: Blocked by device registry
- Application: Blocked by auto-discovery quarantine list

### ❓ Unknown Devices (23 devices)

**Extended range to discover:** 0x8054-0x8059, 0x805B-0x806B

These devices require hardware-based discovery on actual Dell Latitude 5450 MIL-SPEC platform.

---

## Enumeration Tools

### 1. Device Capability Extractor

**File:** `scripts/extract-device-capabilities.py`

Extracts all methods, registers, and capabilities from device implementations.

```bash
# Extract capabilities from all devices
python3 scripts/extract-device-capabilities.py

# Verbose output
python3 scripts/extract-device-capabilities.py --verbose

# Custom output file
python3 scripts/extract-device-capabilities.py --output custom.json
```

**Output:** `DSMIL_DEVICE_CAPABILITIES.json` (222 KB)
- Complete catalog of 656 operations
- 273 hardware registers mapped
- Device complexity analysis

### 2. Documentation Generator

**File:** `scripts/generate-device-docs.py`

Generates individual markdown documentation for each device.

```bash
# Generate documentation for all devices
python3 scripts/generate-device-docs.py

# Custom output directory
python3 scripts/generate-device-docs.py --output-dir custom/docs

# Verbose output
python3 scripts/generate-device-docs.py --verbose
```

**Output:** `00-documentation/devices/` (81 markdown files)
- 80 individual device documentation files
- 1 comprehensive index/README
- Operation signatures, register maps, usage examples

### 3. DSMIL Command-Line Interface

**File:** `scripts/dsmil-cli.py`

Comprehensive CLI for device enumeration and exploration.

See [Command-Line Interface](#command-line-interface) section for detailed usage.

### 4. Interactive Probe

**File:** `scripts/interactive-probes/04_test_dsmil_enumeration.py`

Interactive TUI for device enumeration on actual hardware.

See [Interactive Probes](#interactive-probes) section for detailed usage.

---

## Command-Line Interface

### Installation

No installation required. Uses only Python standard library.

### Commands

#### `list` - List All Devices

```bash
python3 scripts/dsmil-cli.py list
```

**Output:**
```
ID         Risk Name                                Ops   Regs  Group
----------------------------------------------------------------------------------------------------
0x8000     🟡   TPMControlDevice                     41    8     Group 0: Core Security
0x8001     🟡   BootSecurityDevice                   11    8     Group 0: Core Security
0x8002     🟡   CredentialVaultDevice                15    8     Group 0: Core Security
...
Total: 80 devices | 🟢 SAFE  🟡 MONITORED  🔴 QUARANTINED
```

#### `info` - Show Device Information

```bash
python3 scripts/dsmil-cli.py info 0x8000
```

**Output:**
```
================================================================================
  DEVICE 0x8000: TPMControlDevice
================================================================================

📋 Device Information:
  ID: 0x8000 (32768)
  Name: TPMControlDevice
  Group: Group 0: Core Security
  Risk Level: 🟡 MONITORED
  File: device_0x8000_tpm_control.py

📊 Statistics:
  Operations: 41
  Registers: 8
  Private Methods: 4

⚙️  Operations (showing first 10):
   1. initialize()
   2. get_capabilities()
   3. get_status()
   4. read_register(register)
   5. generate_key(algorithm, key_size)
   ...
```

#### `operations` - List Device Operations

```bash
# List all operations
python3 scripts/dsmil-cli.py operations 0x8000

# Include operation descriptions
python3 scripts/dsmil-cli.py operations 0x8000 --verbose
```

#### `search` - Search Devices

```bash
# Search by name
python3 scripts/dsmil-cli.py search "TPM"

# Search by description
python3 scripts/dsmil-cli.py search "security"

# Search by device ID
python3 scripts/dsmil-cli.py search "8000"
```

#### `stats` - Show Statistics

```bash
python3 scripts/dsmil-cli.py stats
```

**Output:**
```
📊 Overall Statistics:
  Total Devices: 80
  Total Operations: 656
  Total Registers: 273
  Average Operations per Device: 8.2

📦 Devices by Group:
  Group 0: Core Security: 9 devices
  Group 1: Extended Security: 12 devices
  ...

🏆 Most Complex Devices:
  1. 0x8000: TPMControlDevice (41 operations)
  2. 0x8002: CredentialVaultDevice (15 operations)
  ...

🔒 Risk Level Breakdown:
  🟢 SAFE: 75 devices
  🟡 MONITORED: 5 devices
  🔴 QUARANTINED: 0 devices
```

#### `groups` - List Devices by Group

```bash
python3 scripts/dsmil-cli.py groups
```

#### `complex` - Show Most Complex Devices

```bash
# Top 10 (default)
python3 scripts/dsmil-cli.py complex

# Top 5
python3 scripts/dsmil-cli.py complex --top 5
```

#### `export` - Export Device List

```bash
# Export to CSV
python3 scripts/dsmil-cli.py export --format csv > devices.csv

# Export to JSON
python3 scripts/dsmil-cli.py export --format json > devices.json

# Export to Markdown table
python3 scripts/dsmil-cli.py export --format markdown > devices.md
```

### Advanced Usage

```bash
# Use custom capability file
python3 scripts/dsmil-cli.py --cap-file custom.json list

# Chain commands
python3 scripts/dsmil-cli.py search "security" | grep "MONITORED"

# Export and analyze
python3 scripts/dsmil-cli.py export --format csv | \
  awk -F, '{print $1, $3}' | \
  sort -k2 -nr | head -10
```

---

## Device Documentation

### Location

`00-documentation/devices/`

### Index

**File:** `00-documentation/devices/README.md`

Complete index of all 80 devices organized by:
- Most complex devices
- Device groups
- Risk levels

### Individual Device Documentation

**Format:** `00-documentation/devices/<DEVICE_ID>.md`

Each device has comprehensive documentation including:

1. **Device Information**
   - Device ID (hex and decimal)
   - Name and description
   - Group classification
   - Risk level
   - Total operations and registers
   - Implementation file

2. **Operations**
   - Core operations (initialize, get_status, etc.)
   - Configuration operations
   - Advanced operations
   - Full method signatures
   - Parameter descriptions
   - Return types

3. **Hardware Registers**
   - Register name
   - Constant definition
   - Offset address

4. **Device Constants**
   - Status bits
   - Capability flags
   - Command codes
   - Error codes
   - Operating modes

5. **Usage Examples**
   - Python code snippets
   - Initialization sequence
   - Common operations

6. **Safety Warnings**
   - Risk level explanation
   - READ/WRITE operation guidelines
   - Testing recommendations

### Example: TPM Control Device

**File:** `00-documentation/devices/0x8000.md`

```markdown
# Device 0x8000: TPMControlDevice

## Device Information
| Property | Value |
|----------|-------|
| **Device ID** | `0x8000` (32768) |
| **Name** | TPMControlDevice |
| **Group** | Group 0: Core Security (0x8000-0x800B) |
| **Risk Level** | 🟡 MONITORED (85% safe for READ) |
| **Total Operations** | 41 |
| **Total Registers** | 8 |

## Operations (41)
- initialize()
- get_capabilities()
- get_status()
- read_register(register)
- generate_key(algorithm, key_size)
- read_pcr(pcr_index)
- extend_pcr(pcr_index, data)
- seal_data(data, pcr_list)
- unseal_data(sealed_blob)
- get_random(num_bytes)
- create_primary(hierarchy, algorithm)
- create_key(parent_handle, algorithm, attributes)
- load_key(parent_handle, private_blob, public_blob)
- sign(key_handle, data, scheme)
- verify_signature(key_handle, data, signature, scheme)
- quote(key_handle, pcr_list, nonce)
- ml_kem_keypair(algorithm)
- ml_kem_encapsulate(public_key, algorithm)
- ml_kem_decapsulate(secret_key, ciphertext, algorithm)
- ml_dsa_keypair(algorithm)
- ml_dsa_sign(secret_key, data, algorithm)
- ml_dsa_verify(public_key, data, signature, algorithm)
- pqc_encrypt(data, recipient_ml_kem_pubkey)
- pqc_decrypt(kem_ciphertext, data_ciphertext, nonce, tag, secret_key)
- validate_pqc_compliance(kem, signature, symmetric, hash_algo)
- get_pqc_status()
...
```

### Browsing Documentation

```bash
# View index
cat 00-documentation/devices/README.md

# View specific device
cat 00-documentation/devices/0x8000.md

# Search across all documentation
grep -r "Post-Quantum" 00-documentation/devices/

# List all MONITORED devices
grep -l "MONITORED" 00-documentation/devices/*.md

# Find devices with most operations
grep "Total Operations" 00-documentation/devices/*.md | \
  sort -t'|' -k4 -nr | head -10
```

---

## Interactive Probes

### DSMIL Enumeration Probe

**File:** `scripts/interactive-probes/04_test_dsmil_enumeration.py`

**⚠️ Requires:** Dell Latitude 5450 MIL-SPEC hardware with DSMIL driver loaded

### Running the Probe

```bash
# On Dell Latitude 5450 MIL-SPEC hardware
cd ~/LAT5150DRVMIL/scripts/interactive-probes
python3 04_test_dsmil_enumeration.py
```

### Menu Options

```
╔══════════════════════════════════════════════════════════════════════════╗
║          DSMIL ENUMERATION INTERACTIVE PROBE                             ║
║          Dell Latitude 5450 MIL-SPEC Edition                             ║
╚══════════════════════════════════════════════════════════════════════════╝

Dell System Military Integration Layer (DSMIL):
  • 108 total DSMIL devices on platform
  • 80 devices FULLY IMPLEMENTED (74.1% coverage)
  • 5 devices QUARANTINED for safety (4.6%)
  • 23 devices UNKNOWN in extended range (21.3%)
  • 6 military tokens (0x049e-0x04a3)
  • Security levels: UNCLASSIFIED → TOP_SECRET

1. Check DSMIL driver status
2. Check DSMIL device nodes
3. Check military token status
4. List implemented devices (80/108)
5. Launch DSMIL interactive menu (full TUI)
6. Run device discovery
7. Probe specific device
8. Show quick reference
9. Run all checks
0. Exit
```

### Features

1. **Driver Status Check**
   - Verify DSMIL driver is loaded
   - Check driver version
   - Display module information

2. **Device Node Detection**
   - Scan for `/dev/dsmil*` device nodes
   - Check permissions and ownership
   - Verify accessibility

3. **Military Token Validation**
   - Enumerate 6 military tokens (0x049e-0x04a3)
   - Check security levels (UNCLASSIFIED → TOP_SECRET)
   - Verify token accessibility

4. **Device Listing**
   - Display all 80 implemented devices
   - Show risk levels and groups
   - Full statistics (80 active + 5 quarantined + 23 unknown)

5. **DSMIL Interactive Menu**
   - Launch full TUI (`02-tools/dsmil-devices/dsmil_menu.py`)
   - Complete device control interface
   - Real-time device status monitoring

6. **Device Discovery**
   - Run `dsmil_discover.py` with summary
   - Hardware-based device scanning
   - Automated device enumeration

7. **Device Probing**
   - Probe specific device by ID
   - Read-only safe operations
   - Detailed device response analysis

8. **Quick Reference**
   - Command-line tool usage
   - Driver loading instructions
   - Military token information
   - Device group descriptions

### Prerequisites

**On Dell Latitude 5450 MIL-SPEC hardware:**

```bash
# 1. Load DSMIL driver
sudo modprobe dsmil-72dev
# Or: sudo insmod /path/to/dsmil-72dev.ko

# 2. Verify driver loaded
lsmod | grep dsmil

# 3. Check device nodes
ls -la /dev/dsmil*

# 4. Run probe
python3 scripts/interactive-probes/04_test_dsmil_enumeration.py
```

---

## Discovery Strategy

### Overview

Complete roadmap to discover the remaining 23 unknown devices (0x8054-0x806B).

**Document:** `00-documentation/DSMIL_DISCOVERY_STRATEGY.md`

### 3-Phase Plan

#### Phase 1: Capability Documentation ✅ **COMPLETE**

**Duration:** 2-3 days
**Status:** ✅ Done

**Deliverables:**
- ✅ Device capability extraction tool
- ✅ JSON capability catalog (656 operations)
- ✅ Individual markdown documentation (81 files)
- ✅ DSMIL CLI tool
- ⏭️ Next: Interactive capability browser TUI

#### Phase 2: Extended Device Discovery

**Duration:** 4-6 weeks
**Status:** ⏸️ Awaiting hardware deployment
**⚠️ REQUIRES:** Dell Latitude 5450 MIL-SPEC hardware

**6 Discovery Methods:**

1. **ACPI Table Analysis**
   - Scan DSDT/SSDT tables for device references
   - Extract device names and register offsets
   - Identify WMI methods

2. **SMBIOS Token Scanning**
   - Query Dell military tokens (0x049e-0x04a3)
   - Scan SMBIOS for device activation flags
   - Check feature requirements

3. **Kernel Driver Analysis**
   - Load dell-smbios, dell-wmi modules
   - Monitor WMI events
   - Analyze driver parameters

4. **Intel ME Investigation**
   - Check Management Engine interfaces
   - Scan for HECI/MEI devices
   - Identify ME-controlled features

5. **MSR Register Scanning**
   - Scan Model-Specific Registers (0x800-0x8FF)
   - Check for device activation MSRs
   - Identify hardware feature flags

6. **Direct Hardware Probing**
   - Systematic read-only probing (0x8054-0x806B)
   - Safe, monitored operations only
   - Response analysis and logging

**Safety Protocol:**
- READ-ONLY operations during discovery
- No WRITE until device is understood
- Automatic quarantine of destructive devices
- Comprehensive logging

#### Phase 3: Integration

**Duration:** 1-2 weeks (after Phase 2)
**Status:** ⏸️ Pending Phase 2 completion

**Tasks:**
1. Create device implementation files
2. Add to quarantine list (if needed)
3. Update auto-discovery system
4. Comprehensive testing
5. Documentation updates

**Goal:** 100% device coverage (108/108)

### Discovery Tools (Future)

**To be created for Phase 2:**

1. **hardware-discovery-scanner.py**
   - Automated ACPI/SMBIOS/WMI/ME/MSR scanning
   - Device candidate identification
   - Metadata extraction

2. **safe-device-prober.py**
   - Read-only probing with safety checks
   - Response logging
   - Behavior analysis

3. **generate-device-from-discovery.py**
   - Generate device implementation from discovery data
   - Auto-generate device files
   - Safety validation

---

## Safety & Security

### Critical Safety Rules

**NEVER ACCESS WITHOUT AUTHORIZATION:**
```
🔴 0x8009 - DATA DESTRUCTION      (DOD-level data wipe)
🔴 0x800A - CASCADE WIPE          (Secondary wipe system)
🔴 0x800B - HARDWARE SANITIZE     (Physical destruction trigger)
🔴 0x8019 - NETWORK KILL          (Network interface destruction)
🔴 0x8029 - COMMS BLACKOUT        (Communications kill switch)
```

### Protection Layers

**4-Layer Quarantine System:**

1. **Hardware Layer**
   - Devices cannot be accessed via SMI/ACPI
   - BIOS-level protection

2. **Kernel Layer**
   - Blocked by device drivers
   - No /dev node creation

3. **Software Layer**
   - Blocked by device registry
   - Auto-discovery skips quarantined devices

4. **Application Layer**
   - Quarantine list enforcement
   - Explicit checks in all tools

### Discovery Safety Protocol

**Level 1: Read-Only Probing**
- ✅ Only READ operations allowed
- ✅ No WRITE, no EXECUTE, no CONFIGURE
- ✅ Automatic abort on error
- ✅ Full logging

**Level 2: Controlled Testing**
- ⚠️ READ and STATUS operations
- ⚠️ Limited CONFIGURE operations
- ⚠️ Requires dual authorization
- ⚠️ Full monitoring

**Level 3: Full Integration**
- 🔒 All operations available
- 🔒 Complete safety review required
- 🔒 Quarantine enforcement active
- 🔒 Post-integration testing

### Quarantine Detection Rules

**Automatically quarantine device if:**
1. Name contains: DESTRUCT, WIPE, KILL, SANITIZE, BLACKOUT
2. Has WRITE-ONLY registers with no readback
3. Triggers hardware resets during probing
4. Requires military authorization tokens
5. Has irreversible operations

### Risk Levels

**🟢 SAFE (75 devices)**
- Standard operations
- Low risk
- READ and WRITE operations generally safe
- Normal testing procedures

**🟡 MONITORED (5 devices)**
- Security-critical
- READ operations 85% safe
- WRITE operations require careful review
- Test in safe environment first
- Devices: 0x8000, 0x8001, 0x8002, 0x8007, 0x8008

**🔴 QUARANTINED (5 devices)**
- Destructive operations
- Permanently blocked at all levels
- NEVER ACCESS without explicit authorization
- Devices: 0x8009, 0x800A, 0x800B, 0x8019, 0x8029

---

## Next Steps

### Immediate Actions (No Hardware Required)

✅ **Completed:**
1. ✅ Device capability extraction
2. ✅ JSON capability catalog
3. ✅ Individual device documentation
4. ✅ DSMIL CLI tool

📋 **Recommended Next:**
1. Create interactive capability browser TUI
2. Generate additional analysis reports
3. Create device comparison tools
4. Build device dependency graph

### Hardware Deployment (Requires Dell Latitude 5450 MIL-SPEC)

**Phase 2 Tasks:**
1. Deploy framework to actual hardware
2. Load DSMIL driver
3. Run interactive probe (04_test_dsmil_enumeration.py)
4. Execute 6 discovery methods
5. Identify 23 unknown devices
6. Create safety assessments
7. Implement safe devices
8. Quarantine dangerous devices
9. Achieve 100% coverage (108/108)

**Timeline:** 4-6 weeks on actual hardware

### Resources

**Documentation:**
- `00-documentation/DSMIL_DISCOVERY_STRATEGY.md` - Complete discovery roadmap
- `00-documentation/devices/README.md` - Device documentation index
- `02-tools/dsmil-devices/COMPLETE_DEVICE_DISCOVERY.md` - Current implementation status

**Tools:**
- `scripts/dsmil-cli.py` - Command-line interface
- `scripts/extract-device-capabilities.py` - Capability extractor
- `scripts/generate-device-docs.py` - Documentation generator
- `scripts/interactive-probes/04_test_dsmil_enumeration.py` - Interactive probe

**Data:**
- `DSMIL_DEVICE_CAPABILITIES.json` - Complete capability catalog (222 KB)

**Implementation:**
- `02-tools/dsmil-devices/devices/` - 80 device implementation files
- `02-tools/dsmil-devices/dsmil_auto_discover.py` - Auto-discovery framework
- `02-tools/dsmil-devices/lib/device_registry.py` - Device registry

---

## Statistics Summary

### Device Coverage

```
Total Devices: 108
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ACTIVE:       80 devices (74.1%)
🔴 QUARANTINED:   5 devices ( 4.6%)
❓ UNKNOWN:      23 devices (21.3%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Capability Statistics

```
Total Operations:     656
Total Registers:      273
Average Ops/Device:   8.2
Most Complex Device:  TPM Control (41 operations)
```

### Group Coverage

```
Group 0: Core Security        →  9/12 devices (100% standard, 3 quarantined)
Group 1: Extended Security    → 12/12 devices (100%)
Group 2: Network/Comms        → 11/12 devices (100% standard, 1 quarantined)
Group 3: Data Processing      → 11/12 devices (100% standard, 1 quarantined)
Group 4: Storage Management   → 12/12 devices (100%)
Group 5: Peripheral Control   → 12/12 devices (100%)
Group 6: Training/Simulation  → 12/12 devices (100%)
Extended Range                →  1/24 devices (4%)
```

### Risk Level Distribution

```
🟢 SAFE:         75 devices (93.8%)
🟡 MONITORED:     5 devices ( 6.2%)
🔴 QUARANTINED:   5 devices (not in active count)
```

---

## Support & Feedback

### Reporting Issues

For issues, bugs, or feature requests:
- GitHub: https://github.com/anthropics/claude-code/issues
- Include: Framework version, platform details, reproduction steps

### Contributing

To contribute device implementations or documentation:
1. Follow existing device template
2. Include comprehensive safety assessment
3. Add device to auto-discovery system
4. Update documentation
5. Submit for review

### Security Concerns

For security-related concerns about quarantined devices or safety protocols:
- Contact: Security team
- Include: Device ID, operation attempted, context
- **DO NOT** attempt to bypass quarantine protections

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Framework Version:** 2.0.0 (Auto-Discovery)
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Platform:** Dell Latitude 5450 MIL-SPEC JRTC1 Training Variant

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    DSMIL ENUMERATION QUICK REFERENCE                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  LIST ALL DEVICES:                                                       ║
║    python3 scripts/dsmil-cli.py list                                     ║
║                                                                          ║
║  DEVICE INFO:                                                            ║
║    python3 scripts/dsmil-cli.py info 0x8000                              ║
║                                                                          ║
║  SEARCH DEVICES:                                                         ║
║    python3 scripts/dsmil-cli.py search "security"                        ║
║                                                                          ║
║  SHOW STATISTICS:                                                        ║
║    python3 scripts/dsmil-cli.py stats                                    ║
║                                                                          ║
║  INTERACTIVE PROBE (on actual hardware):                                 ║
║    python3 scripts/interactive-probes/04_test_dsmil_enumeration.py       ║
║                                                                          ║
║  DOCUMENTATION:                                                          ║
║    cat 00-documentation/devices/README.md                                ║
║    cat 00-documentation/devices/0x8000.md                                ║
║                                                                          ║
║  EXPORT TO CSV:                                                          ║
║    python3 scripts/dsmil-cli.py export --format csv > devices.csv        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  COVERAGE: 80/108 devices (74.1%) | 656 operations | 273 registers      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

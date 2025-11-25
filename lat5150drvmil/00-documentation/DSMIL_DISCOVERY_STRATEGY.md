# DSMIL Device Discovery and Documentation Strategy

**Target Platform:** Dell Latitude 5450 MIL-SPEC
**Current Status:** 80/108 devices (74.1%)
**Remaining:** 23 unknown devices (0x8054-0x806B)
**Created:** 2025-11-08

---

## Executive Summary

### Current State
- **80 devices ACTIVE** (74.1% coverage) - Fully implemented
- **5 devices QUARANTINED** (4.6%) - Permanently blocked for safety
- **23 devices UNKNOWN** (21.3%) - Extended range 0x8054-0x806B
- **Device capabilities**: Partially documented (need comprehensive catalog)

### Objectives
1. **Discover and integrate** the 23 unknown devices in extended range
2. **Document all capabilities** for all 108 devices (methods, registers, operations)
3. **Create safety framework** to prevent accidental triggering of destructive devices
4. **Build automated discovery pipeline** for hardware-based enumeration

---

## Challenge: Extended Range Devices (0x8054-0x806B)

### What We Know
```
0x8054-0x8059 (6 devices)  - Unknown
0x805A         (1 device)  - SensorArray (INTEGRATED)
0x805B-0x8063 (9 devices)  - Unknown
0x8064         (1 device)  - Unknown
0x8065-0x806B (7 devices)  - Unknown
```

### Why These Are Unknown
1. **Not in standard grid** - Beyond the 7-group × 12-device standard layout
2. **No Dell documentation** - Not covered in standard DSMIL specs
3. **Specialized features** - Likely advanced military-specific capabilities
4. **Hardware-specific** - May only be accessible on actual Dell Latitude 5450 hardware

### Discovery Methods Required

#### Method 1: ACPI Table Analysis
**What:** Scan ACPI DSDT/SSDT tables for device references

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Extract ACPI tables
sudo acpidump > acpi_tables.dat
acpixtract -a acpi_tables.dat

# Search for DSMIL device references
iasl -d DSDT.dat
grep -i "8054\|8055\|8056\|8057\|8058\|8059\|805B\|805C" DSDT.dsl

# Look for WMI methods
grep -i "WMAA\|WMAB\|WMAC" DSDT.dsl
```

**Expected Discoveries:**
- Device names and descriptions
- Register offsets and access methods
- Dependencies and initialization sequences

#### Method 2: SMBIOS Token Scanning
**What:** Scan Dell SMBIOS tokens for hidden devices

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Dell SMBIOS token scanner
sudo dmidecode --type 0,1,2,3,11,12,14

# Search for military tokens (0x049e-0x04a3)
sudo python3 <<EOF
import subprocess
for token in range(0x049e, 0x04a4):
    # Use Dell WMI interface to query token
    # (Requires dell-smbios-wmi kernel module)
    print(f"Token 0x{token:04X}: [query via WMI]")
EOF
```

**Expected Discoveries:**
- Device enable/disable flags
- Feature activation requirements
- Security level indicators

#### Method 3: Kernel Driver Analysis
**What:** Analyze Dell kernel modules for device references

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Load Dell kernel modules
sudo modprobe dell-smbios
sudo modprobe dell-wmi
sudo modprobe dell-laptop

# Check loaded modules
lsmod | grep dell

# Examine module parameters
sudo modinfo dell-smbios
sudo modinfo dell-wmi

# Search Dell WMI events
sudo dmesg | grep -i "dell\|wmi\|dsmil"

# Monitor WMI events
sudo wmi-bmof -a /sys/bus/wmi/devices/*/bmof
```

**Expected Discoveries:**
- WMI method names and GUIDs
- Device initialization sequences
- Required permissions and capabilities

#### Method 4: Intel Management Engine (ME)
**What:** Check Intel ME for hidden interfaces

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Check Intel ME version
sudo intelmetool -m

# Look for HECI/MEI interfaces
ls -la /dev/mei*

# Check for military-specific ME extensions
sudo lspci -vv | grep -A 10 "Management Engine"
```

**Expected Discoveries:**
- ME-controlled device interfaces
- Hardware security features
- Attestation capabilities

#### Method 5: MSR Register Scanning
**What:** Scan Model-Specific Registers for device-specific MSRs

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Load MSR module
sudo modprobe msr

# Scan for DSMIL MSRs (requires identification first)
# Example: Check for NPU MSRs
sudo rdmsr 0x1A0  # IA32_MISC_ENABLE
sudo rdmsr 0x1FC  # IA32_POWER_CTL

# Check for undocumented MSRs in 0x800-0x8FF range
for msr in $(seq 0x800 0x8FF); do
    sudo rdmsr $msr 2>/dev/null && echo "MSR 0x${msr}: Found"
done
```

**Expected Discoveries:**
- Hardware feature flags
- Device activation registers
- Performance monitoring capabilities

#### Method 6: Direct Hardware Probing
**What:** Use DSMIL probe tool with systematic scanning

**On Dell Latitude 5450 MIL-SPEC:**
```bash
# Safe probing with read-only operations
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-devices

# Probe each unknown device
for dev_id in 0x8054 0x8055 0x8056 0x8057 0x8058 0x8059 \
              0x805B 0x805C 0x805D 0x805E 0x805F 0x8060 \
              0x8061 0x8062 0x8063 0x8064 0x8065 0x8066 \
              0x8067 0x8068 0x8069 0x806A 0x806B; do
    echo "=== Probing $dev_id ==="
    sudo python3 dsmil_probe.py $dev_id --read-only --safe-mode
done
```

**Safety Protocol:**
- READ-ONLY operations only
- No WRITE operations during discovery
- Quarantine any destructive devices immediately
- Log all responses for analysis

---

## Challenge: Document All Device Capabilities

### Current Documentation Gap

Looking at device_0x8000_tpm_control.py (TPM Control), we can see it has **40+ operations**:

**Standard Operations:**
- `initialize()`, `get_status()`, `read_register()`
- `generate_key()`, `read_pcr()`, `extend_pcr()`
- `seal_data()`, `unseal_data()`, `get_random()`

**TPM 2.0 Operations:**
- `create_primary()`, `create_key()`, `load_key()`, `flush_context()`
- `evict_control()`, `sign()`, `verify_signature()`
- `hash()`, `hmac()`, `quote()`, `activate_credential()`
- `certify()`, `nv_define()`, `nv_write()`, `nv_read()`
- `get_capability()`, `clear_tpm()`, `reset_pcr()`

**Post-Quantum Cryptography:**
- `ml_kem_keypair()`, `ml_kem_encapsulate()`, `ml_kem_decapsulate()`
- `ml_dsa_keypair()`, `ml_dsa_sign()`, `ml_dsa_verify()`
- `hybrid_sign()`, `hybrid_verify()`
- `pqc_encrypt()`, `pqc_decrypt()`
- `validate_pqc_compliance()`, `get_pqc_status()`

**But:** Other devices may have similar complexity, and we don't have a complete catalog.

### Documentation Strategy

#### Step 1: Automated Capability Extraction
Create a tool to extract all methods from device implementations:

```python
# scripts/extract-device-capabilities.py
import os
import ast
import json
from pathlib import Path

def extract_device_capabilities(device_file):
    """Extract all methods from a device implementation"""
    with open(device_file, 'r') as f:
        tree = ast.parse(f.read())

    capabilities = {
        "device_id": None,
        "name": None,
        "registers": [],
        "operations": [],
        "constants": {}
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                # Extract methods
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith('_'):
                        capabilities["operations"].append({
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "docstring": ast.get_docstring(item)
                        })

                # Extract register constants
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            if target.id.startswith('REG_'):
                                capabilities["registers"].append(target.id)

    return capabilities

# Scan all device files
device_dir = Path("02-tools/dsmil-devices/devices")
all_capabilities = {}

for device_file in device_dir.glob("device_0x*.py"):
    dev_id = device_file.stem.split('_')[1]
    all_capabilities[dev_id] = extract_device_capabilities(device_file)

# Save to JSON
with open("DSMIL_DEVICE_CAPABILITIES.json", 'w') as f:
    json.dump(all_capabilities, f, indent=2)

print(f"Extracted capabilities for {len(all_capabilities)} devices")
```

#### Step 2: Generate Comprehensive Documentation
Create markdown documentation from extracted capabilities:

```python
# scripts/generate-device-docs.py
import json

def generate_device_documentation(capabilities):
    """Generate markdown documentation for each device"""

    for dev_id, caps in capabilities.items():
        doc_path = f"00-documentation/devices/{dev_id}.md"

        with open(doc_path, 'w') as f:
            f.write(f"# Device {dev_id}: {caps['name']}\\n\\n")
            f.write(f"## Overview\\n\\n")
            f.write(f"Device ID: {dev_id}\\n")
            f.write(f"Group: {caps.get('group')}\\n")
            f.write(f"Risk Level: {caps.get('risk_level')}\\n\\n")

            f.write(f"## Operations ({len(caps['operations'])})\\n\\n")
            for op in caps['operations']:
                f.write(f"### `{op['name']}({', '.join(op['args'])})`\\n\\n")
                if op['docstring']:
                    f.write(f"{op['docstring']}\\n\\n")

            f.write(f"## Registers ({len(caps['registers'])})\\n\\n")
            for reg in caps['registers']:
                f.write(f"- `{reg}`\\n")
```

#### Step 3: Interactive Capability Browser
Create a TUI to browse device capabilities:

```python
# scripts/browse-device-capabilities.py
import curses
import json

def browse_capabilities(stdscr, capabilities):
    """Interactive TUI to browse device capabilities"""

    # List devices
    devices = list(capabilities.keys())
    current_device = 0
    current_operation = 0

    while True:
        stdscr.clear()

        # Show current device
        dev_id = devices[current_device]
        caps = capabilities[dev_id]

        stdscr.addstr(0, 0, f"Device {dev_id}: {caps['name']}")
        stdscr.addstr(1, 0, "=" * 80)

        # Show operations
        stdscr.addstr(3, 0, f"Operations ({len(caps['operations'])}):")
        for i, op in enumerate(caps['operations']):
            if i == current_operation:
                stdscr.addstr(5 + i, 2, f"> {op['name']}", curses.A_REVERSE)
            else:
                stdscr.addstr(5 + i, 2, f"  {op['name']}")

        # Navigation
        key = stdscr.getch()
        if key == curses.KEY_UP:
            current_operation = max(0, current_operation - 1)
        elif key == curses.KEY_DOWN:
            current_operation = min(len(caps['operations']) - 1, current_operation + 1)
        elif key == curses.KEY_LEFT:
            current_device = max(0, current_device - 1)
            current_operation = 0
        elif key == curses.KEY_RIGHT:
            current_device = min(len(devices) - 1, current_device + 1)
            current_operation = 0
        elif key == ord('q'):
            break
```

---

## Implementation Timeline

### Phase 1: Capability Documentation (2-3 days)
**Goal:** Document all 80 integrated devices

**Tasks:**
1. Create `scripts/extract-device-capabilities.py`
2. Run extraction on all 80 device files
3. Generate `DSMIL_DEVICE_CAPABILITIES.json`
4. Create individual markdown docs per device
5. Build interactive capability browser TUI

**Deliverables:**
- `DSMIL_DEVICE_CAPABILITIES.json` (complete capability catalog)
- `00-documentation/devices/*.md` (80 device documentation files)
- `scripts/browse-device-capabilities.py` (interactive browser)

### Phase 2: Extended Device Discovery (4-6 weeks)
**Goal:** Discover and integrate 23 unknown devices

**⚠️ REQUIRES DELL LATITUDE 5450 MIL-SPEC HARDWARE ⚠️**

**Week 1-2: ACPI/SMBIOS Analysis**
1. Extract ACPI tables on actual hardware
2. Scan SMBIOS tokens (0x049e-0x04a3)
3. Analyze Dell WMI methods
4. Document findings

**Week 3-4: Kernel and ME Analysis**
1. Load Dell kernel modules
2. Monitor WMI events
3. Check Intel ME interfaces
4. Scan MSR registers

**Week 5-6: Hardware Probing**
1. Systematic read-only probes (0x8054-0x806B)
2. Analyze responses
3. Identify device types
4. Create device implementations

**Safety Protocol:**
- All probing in READ-ONLY mode
- No WRITE operations until device is understood
- Quarantine any destructive devices immediately
- Log all operations for safety review

**Deliverables:**
- Device type identification for all 23 devices
- Safety assessment for each device
- Implementation files for safe devices
- Quarantine list for dangerous devices

### Phase 3: Integration and Testing (1-2 weeks)
**Goal:** Integrate discovered devices into framework

**Tasks:**
1. Create device implementation files for safe devices
2. Add to quarantine list for dangerous devices
3. Update auto-discovery system
4. Comprehensive testing on actual hardware
5. Update documentation

**Deliverables:**
- New device files in `02-tools/dsmil-devices/devices/`
- Updated quarantine list in safety library
- Updated `COMPLETE_DEVICE_DISCOVERY.md`
- Test results showing 100% coverage

---

## Safety Framework

### Critical Safety Rules

**NEVER ACCESS WITHOUT AUTHORIZATION:**
```
0x8009 - DATA DESTRUCTION      (DOD-level data wipe)
0x800A - CASCADE WIPE          (Secondary wipe system)
0x800B - HARDWARE SANITIZE     (Physical destruction trigger)
0x8019 - NETWORK KILL          (Network interface destruction)
0x8029 - COMMS BLACKOUT        (Communications kill switch)
```

### Discovery Safety Protocol

**Level 1: Read-Only Probing**
- Only READ operations allowed
- No WRITE, no EXECUTE, no CONFIGURE
- Log all responses
- Automatic abort on error

**Level 2: Controlled Testing**
- READ and STATUS operations
- Limited CONFIGURE operations
- Requires dual authorization
- Full logging and monitoring

**Level 3: Full Integration**
- All operations available
- Requires complete safety review
- Quarantine enforcement active
- Post-integration testing

### Quarantine Detection Rules

**Automatically quarantine if device:**
1. Has "DESTRUCT", "WIPE", "KILL", "SANITIZE", "BLACKOUT" in name
2. Has WRITE-ONLY registers with no readback
3. Triggers hardware resets during probing
4. Requires military authorization tokens
5. Has irreversible operations

---

## Tools to Create

### 1. Device Capability Extractor
**File:** `scripts/extract-device-capabilities.py`
**Purpose:** Parse all device implementations and extract capabilities
**Output:** `DSMIL_DEVICE_CAPABILITIES.json`

### 2. Hardware Discovery Scanner
**File:** `scripts/hardware-discovery-scanner.py`
**Purpose:** Automated ACPI/SMBIOS/WMI/ME/MSR scanning
**Output:** Device candidates with metadata
**Requires:** Dell Latitude 5450 MIL-SPEC hardware

### 3. Safe Device Prober
**File:** `scripts/safe-device-prober.py`
**Purpose:** Systematic read-only probing with safety checks
**Output:** Device responses and behavior logs
**Requires:** Dell Latitude 5450 MIL-SPEC hardware

### 4. Capability Browser TUI
**File:** `scripts/browse-device-capabilities.py`
**Purpose:** Interactive browser for device capabilities
**Output:** Interactive TUI interface

### 5. Device Generator (Enhanced)
**File:** `scripts/generate-device-from-discovery.py`
**Purpose:** Generate device implementation from discovery data
**Output:** Complete device implementation file

---

## Expected Results

### Short Term (Phase 1 - 2-3 days)
- ✅ Complete capability catalog for 80 devices
- ✅ Individual documentation for each device
- ✅ Interactive capability browser
- ✅ Developer reference guide

### Medium Term (Phase 2 - 4-6 weeks)
- ✅ Discovery of 23 unknown devices
- ✅ Device type identification
- ✅ Safety assessment complete
- ✅ Integration plan for safe devices

### Long Term (Phase 3 - 1-2 weeks after Phase 2)
- ✅ **100% device coverage** (108/108 devices)
- ✅ All safe devices integrated
- ✅ All dangerous devices quarantined
- ✅ Complete device documentation
- ✅ Production-ready DSMIL framework

---

## Risk Assessment

### Low Risk
- ✅ Phase 1 (Documentation) - No hardware access required
- ✅ Read-only probing with safety checks
- ✅ ACPI/SMBIOS analysis (passive)

### Medium Risk
- ⚠️ WMI method calls (can trigger device state changes)
- ⚠️ MSR register reads (can expose sensitive data)
- ⚠️ Intel ME interaction (security implications)

### High Risk
- 🔴 ANY write operations to unknown devices
- 🔴 Executing methods without understanding behavior
- 🔴 Disabling quarantine protections
- 🔴 Testing destructive devices

### Mitigation Strategies
1. **Phased approach** - Documentation first, hardware discovery only on actual platform
2. **Read-only first** - Never write until device is fully understood
3. **Safety checks** - Multiple layers of quarantine enforcement
4. **Logging** - Comprehensive logging of all operations
5. **Dual authorization** - Require explicit approval for risky operations

---

## Success Metrics

**Phase 1 Complete:**
- [ ] 80 device capability files generated
- [ ] JSON capability catalog created
- [ ] Interactive browser working
- [ ] Documentation complete

**Phase 2 Complete:**
- [ ] All 23 devices identified
- [ ] Device types documented
- [ ] Safety assessment passed
- [ ] No accidental destructive operations

**Phase 3 Complete:**
- [ ] 100% device coverage (108/108)
- [ ] All safe devices integrated
- [ ] Test success rate ≥ 99%
- [ ] Zero safety incidents

---

## References

### Existing Tools
- `02-tools/dsmil-devices/dsmil_discover.py` - Current discovery tool
- `02-tools/dsmil-devices/dsmil_probe.py` - Device probing tool
- `02-tools/dsmil-devices/dsmil_auto_discover.py` - Auto-discovery framework
- `scripts/interactive-probes/04_test_dsmil_enumeration.py` - Interactive enumeration

### Documentation
- `02-tools/dsmil-devices/COMPLETE_DEVICE_DISCOVERY.md` - Current status
- `02-tools/dsmil-devices/lib/dsmil_safety.py` - Safety library
- `02-tools/dsmil-devices/lib/device_registry.py` - Device registry

### Standards
- Dell SMBIOS/WMI specifications
- Intel ACPI specifications
- TPM 2.0 specification
- FIPS 203/204 (Post-Quantum Cryptography)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Status:** Strategy Document
**Next Action:** Phase 1 implementation (device capability extraction)
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

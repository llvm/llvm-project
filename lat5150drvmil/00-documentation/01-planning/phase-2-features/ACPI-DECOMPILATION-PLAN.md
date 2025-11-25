# Comprehensive ACPI Decompilation and Analysis Plan

## 🎯 **Overview**

The enumeration discovered 144 DSMIL ACPI references and 12 DSMIL devices (DSMIL0D0-DSMIL0DB). This plan outlines how to extract, decompile, and analyze the ACPI tables to discover the hidden MIL-SPEC methods, understand device control mechanisms, and implement proper ACPI integration.

## 📋 **ACPI Discovery Summary**

### From Enumeration
- **144 DSMIL references** in ACPI tables
- **12 DSMIL devices**: DSMIL0D0 through DSMIL0DB
- **DSDT Size**: 550KB (large, complex)
- **24 SSDT tables**: Additional functionality
- **JRTC1 marker**: Likely has ACPI methods

### Expected ACPI Methods
Based on the pattern found:
```
L0BS, L0DI - Device 0 methods
L1BS, L1DI - Device 1 methods
...
LABS, LADI - Device A (10) methods
LBBS, LBDI - Device B (11) methods
```

## 🏗️ **Implementation Plan**

### **Phase 1: ACPI Table Extraction**

#### 1.1 Extract All ACPI Tables
```bash
#!/bin/bash
# extract-acpi-tables.sh

ACPI_DIR="/opt/scripts/milspec/acpi-tables"
mkdir -p "$ACPI_DIR"

echo "Extracting ACPI tables..."

# Extract all tables
sudo acpidump > "$ACPI_DIR/acpi.dump"

# Extract individual tables
cd "$ACPI_DIR"
acpixtract -a acpi.dump

# List extracted tables
echo "Extracted tables:"
ls -la *.dat

# Get table summaries
for table in *.dat; do
    echo "=== $table ==="
    iasl -d "$table" 2>/dev/null
    if [ -f "${table%.dat}.dsl" ]; then
        echo "Decompiled to ${table%.dat}.dsl"
        # Get basic info
        grep -E "DefinitionBlock|Device \(|Method \(" "${table%.dat}.dsl" | head -20
    fi
done
```

#### 1.2 Focus on DSMIL Tables
```bash
# Find DSMIL references
echo "Searching for DSMIL references..."

# Search in DSDT
grep -n "DSMIL" dsdt.dsl > dsmil_references.txt

# Search in all SSDTs
for ssdt in ssdt*.dsl; do
    echo "=== $ssdt ===" >> dsmil_references.txt
    grep -n "DSMIL" "$ssdt" >> dsmil_references.txt
done

# Count references
echo "Total DSMIL references: $(grep -c DSMIL dsmil_references.txt)"

# Extract DSMIL device definitions
grep -A 50 -B 5 "Device (DSMIL" *.dsl > dsmil_devices.txt
```

### **Phase 2: DSMIL Method Analysis**

#### 2.1 Method Discovery Script
```python
#!/usr/bin/env python3
# analyze-dsmil-methods.py

import re
import sys
from collections import defaultdict

def parse_acpi_file(filename):
    """Parse ACPI DSL file for DSMIL methods and devices"""
    
    devices = {}
    methods = defaultdict(list)
    current_device = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Find DSMIL devices
        device_match = re.match(r'Device\s*\(DSMIL(\w+)\)', line)
        if device_match:
            current_device = f"DSMIL{device_match.group(1)}"
            devices[current_device] = {
                'methods': [],
                'resources': [],
                'properties': {}
            }
            print(f"Found device: {current_device}")
        
        # Find methods within device
        if current_device:
            method_match = re.match(r'Method\s*\((\w+),\s*(\d+)', line)
            if method_match:
                method_name = method_match.group(1)
                arg_count = method_match.group(2)
                devices[current_device]['methods'].append({
                    'name': method_name,
                    'args': int(arg_count),
                    'line': i + 1
                })
                methods[method_name].append(current_device)
                print(f"  Method: {method_name} (args: {arg_count})")
        
        # End of device scope
        if line == '}' and current_device:
            # Simple scope tracking - could be improved
            current_device = None
        
        i += 1
    
    return devices, methods

def analyze_dsmil_patterns(devices):
    """Analyze patterns in DSMIL devices"""
    
    print("\n=== DSMIL Pattern Analysis ===")
    
    # Common methods across devices
    all_methods = set()
    for device, info in devices.items():
        for method in info['methods']:
            all_methods.add(method['name'])
    
    print(f"\nUnique methods across all DSMIL devices: {all_methods}")
    
    # Device naming pattern
    print("\n=== Device Analysis ===")
    for device in sorted(devices.keys()):
        print(f"{device}: {len(devices[device]['methods'])} methods")
        
    # Method patterns
    print("\n=== Method Patterns ===")
    standard_methods = ['_INI', '_STA', '_DIS', '_PS0', '_PS3']
    for method in standard_methods:
        count = sum(1 for d in devices.values() 
                   if any(m['name'] == method for m in d['methods']))
        if count > 0:
            print(f"{method}: Present in {count} devices")

# Main analysis
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze-dsmil-methods.py <acpi_file.dsl>")
        sys.exit(1)
    
    devices, methods = parse_acpi_file(sys.argv[1])
    analyze_dsmil_patterns(devices)
    
    # Save results
    with open('dsmil_analysis.txt', 'w') as f:
        f.write("DSMIL Device Analysis\n")
        f.write("===================\n\n")
        for device, info in sorted(devices.items()):
            f.write(f"{device}:\n")
            for method in info['methods']:
                f.write(f"  - {method['name']} ({method['args']} args)\n")
            f.write("\n")
```

### **Phase 3: Method Implementation Discovery**

#### 3.1 Extract Method Bodies
```bash
#!/bin/bash
# extract-dsmil-methods.sh

# Extract specific DSMIL methods
for i in {0..9} A B; do
    device="DSMIL0D${i}"
    echo "=== Extracting methods for $device ==="
    
    # Find device definition
    awk "/Device \($device\)/,/^    \}/" dsdt.dsl > "${device}_full.asl"
    
    # Extract individual methods
    for method in _INI _STA ENBL DSBL L${i}BS L${i}DI; do
        awk "/Method \($method/,/^        \}/" "${device}_full.asl" > "${device}_${method}.asl"
        if [ -s "${device}_${method}.asl" ]; then
            echo "  Found method: $method"
        fi
    done
done
```

#### 3.2 Understand Method Signatures
```c
/* Expected DSMIL ACPI method signatures */

// Standard ACPI methods
_INI() - Initialize device
_STA() - Get device status
_DIS() - Disable device
_PS0() - Power state 0 (full power)
_PS3() - Power state 3 (off)

// DSMIL-specific methods (discovered)
ENBL() - Enable DSMIL device
DSBL() - Disable DSMIL device
QURY() - Query device state
WIPE() - Emergency wipe trigger

// Per-device methods (pattern: L[0-B][BS|DI])
L0BS() - Device 0 Base State
L0DI() - Device 0 Device Information
L1BS() - Device 1 Base State
L1DI() - Device 1 Device Information
...
LABS() - Device A (10) Base State
LADI() - Device A (10) Device Information
LBBS() - Device B (11) Base State
LBDI() - Device B (11) Device Information
```

### **Phase 4: ACPI Integration Implementation**

#### 4.1 ACPI Method Caller
```c
/* Add to dell-millspec-enhanced.c */

/* Call DSMIL ACPI methods */
static int milspec_call_dsmil_method(int device_id, const char *method_name,
                                    struct acpi_object_list *args,
                                    struct acpi_buffer *result)
{
    char acpi_path[64];
    acpi_status status;
    acpi_handle handle;
    
    /* Build ACPI path */
    snprintf(acpi_path, sizeof(acpi_path), "\\_SB.DSMIL0D%X.%s",
             device_id, method_name);
    
    /* Get method handle */
    status = acpi_get_handle(NULL, acpi_path, &handle);
    if (ACPI_FAILURE(status)) {
        pr_debug("MIL-SPEC: Method %s not found\n", acpi_path);
        return -ENOENT;
    }
    
    /* Evaluate method */
    status = acpi_evaluate_object(handle, NULL, args, result);
    if (ACPI_FAILURE(status)) {
        pr_err("MIL-SPEC: Failed to call %s: %s\n",
               acpi_path, acpi_format_exception(status));
        return -EIO;
    }
    
    return 0;
}

/* Enable DSMIL device via ACPI */
static int milspec_acpi_enable_dsmil(int device_id)
{
    struct acpi_buffer result = { ACPI_ALLOCATE_BUFFER, NULL };
    union acpi_object *obj;
    int ret;
    
    /* Call ENBL method */
    ret = milspec_call_dsmil_method(device_id, "ENBL", NULL, &result);
    if (ret)
        return ret;
    
    /* Check result */
    obj = result.pointer;
    if (obj && obj->type == ACPI_TYPE_INTEGER) {
        ret = obj->integer.value ? 0 : -EIO;
        pr_info("MIL-SPEC: DSMIL%X ENBL returned %lld\n",
                device_id, obj->integer.value);
    }
    
    ACPI_FREE(result.pointer);
    return ret;
}

/* Query device information */
static int milspec_acpi_query_dsmil_info(int device_id, 
                                        struct dsmil_device_info *info)
{
    struct acpi_buffer result = { ACPI_ALLOCATE_BUFFER, NULL };
    char method_name[8];
    union acpi_object *obj;
    int ret;
    
    /* Build method name L[0-B]DI */
    snprintf(method_name, sizeof(method_name), "L%XDI", device_id);
    
    /* Call device info method */
    ret = milspec_call_dsmil_method(device_id, method_name, NULL, &result);
    if (ret)
        return ret;
    
    /* Parse result buffer */
    obj = result.pointer;
    if (obj && obj->type == ACPI_TYPE_BUFFER) {
        milspec_parse_device_info(obj->buffer.pointer,
                                 obj->buffer.length, info);
    }
    
    ACPI_FREE(result.pointer);
    return 0;
}
```

#### 4.2 ACPI Event Handler
```c
/* ACPI notify handler for DSMIL events */
static void milspec_acpi_notify(acpi_handle handle, u32 event, void *data)
{
    struct milspec_device *mdev = data;
    
    pr_info("MIL-SPEC: ACPI event 0x%x on DSMIL device\n", event);
    
    switch (event) {
    case 0x80: /* Status change */
        milspec_handle_status_change(mdev);
        break;
        
    case 0x81: /* Security alert */
        milspec_handle_security_alert(mdev);
        break;
        
    case 0x82: /* Mode transition */
        milspec_handle_mode_transition(mdev);
        break;
        
    case 0x90: /* JRTC1 training event */
        milspec_handle_jrtc1_event(mdev);
        break;
        
    default:
        pr_warn("MIL-SPEC: Unknown ACPI event 0x%x\n", event);
    }
}

/* Register for ACPI notifications */
static int milspec_register_acpi_notify(void)
{
    acpi_status status;
    int i;
    
    for (i = 0; i < 12; i++) {
        char path[32];
        acpi_handle handle;
        
        snprintf(path, sizeof(path), "\\_SB.DSMIL0D%X", i);
        
        status = acpi_get_handle(NULL, path, &handle);
        if (ACPI_SUCCESS(status)) {
            status = acpi_install_notify_handler(handle,
                                               ACPI_DEVICE_NOTIFY,
                                               milspec_acpi_notify,
                                               &milspec_state.dsmil_devices[i]);
            if (ACPI_SUCCESS(status)) {
                pr_info("MIL-SPEC: Registered ACPI notify for %s\n", path);
            }
        }
    }
    
    return 0;
}
```

### **Phase 5: Hidden ACPI Methods**

#### 5.1 Search for Hidden Methods
```bash
#!/bin/bash
# find-hidden-acpi.sh

echo "Searching for hidden/undocumented ACPI methods..."

# Common military/security related strings
KEYWORDS="WIPE|DEST|SEC|CRYP|AUTH|LOCK|SEAL|BURN|ZERO|CLASSIFIED"

# Search in decompiled ACPI
grep -E "$KEYWORDS" *.dsl | grep -v "^Binary file" > hidden_methods.txt

# Look for interesting bit patterns
grep -E "0xDEAD|0xBEEF|0xCAFE|0x1337|0x8086" *.dsl >> hidden_methods.txt

# Find methods with unusual names
grep "Method (" *.dsl | grep -vE "_[A-Z]{3}|_[A-Z]{2}[0-9]" >> hidden_methods.txt

# JRTC1 related
grep -i "jrtc\|jrot" *.dsl >> hidden_methods.txt

echo "Results saved to hidden_methods.txt"
```

#### 5.2 Reverse Engineer Complex Methods
```python
#!/usr/bin/env python3
# decode-acpi-method.py

def decode_acpi_bytecode(method_body):
    """Decode ACPI bytecode to understand functionality"""
    
    # ACPI opcodes of interest
    opcodes = {
        0x00: "ZeroOp",
        0x01: "OneOp",
        0x0A: "ByteConst",
        0x0B: "WordConst",
        0x0C: "DWordConst",
        0x70: "StoreOp",
        0x71: "RefOfOp",
        0x72: "AddOp",
        0x74: "SubtractOp",
        0x79: "ShiftLeftOp",
        0x7A: "ShiftRightOp",
        0x7B: "AndOp",
        0x7D: "OrOp",
        0x80: "NotOp",
        0x86: "NotifyOp",
        0x92: "CondRefOfOp",
        0x93: "CreateFieldOp",
        0x95: "SleepOp",
        0xA0: "IfOp",
        0xA1: "ElseOp",
        0xA2: "WhileOp",
        0xA4: "ReturnOp",
    }
    
    # Parse and decode
    # ... implementation ...
```

### **Phase 6: Testing and Validation**

#### 6.1 ACPI Method Test Tool
```c
/* Test tool for ACPI methods */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

int test_dsmil_acpi_methods(void)
{
    int fd;
    int device_id;
    
    fd = open("/dev/milspec", O_RDWR);
    if (fd < 0) {
        perror("open");
        return -1;
    }
    
    printf("Testing DSMIL ACPI methods...\n");
    
    /* Test each device */
    for (device_id = 0; device_id < 12; device_id++) {
        struct milspec_acpi_test test = {
            .device_id = device_id,
            .method = "ENBL",
        };
        
        if (ioctl(fd, MILSPEC_IOC_TEST_ACPI, &test) == 0) {
            printf("DSMIL%X: ENBL method exists\n", device_id);
        }
    }
    
    close(fd);
    return 0;
}
```

## 📊 **Implementation Timeline**

### **Week 1: Extraction and Analysis**
- Extract all ACPI tables
- Decompile and analyze DSDT/SSDTs
- Document DSMIL methods

### **Week 2: Method Implementation**
- Implement ACPI method callers
- Add device enumeration
- Test basic functionality

### **Week 3: Hidden Features**
- Search for undocumented methods
- Reverse engineer complex logic
- Implement advanced features

### **Week 4: Integration**
- Integrate with main driver
- Add event handlers
- Create debugging tools

## ⚠️ **Important Considerations**

1. **ACPI Namespace**
   - Methods may be in different scopes
   - Use full paths for reliability
   - Handle missing methods gracefully

2. **Method Arguments**
   - Some methods may require specific arguments
   - Return values may be complex objects
   - Buffer management is critical

3. **Security Methods**
   - WIPE/DEST methods are dangerous
   - Require proper authentication
   - May have hardware interlocks

## 🔍 **Expected Discoveries**

### DSMIL Device Methods
- Activation sequences
- Configuration parameters
- Security interlocks
- Hidden capabilities

### JRTC1 Methods
- Training mode activation
- Safety overrides
- Instructor authentication

### Hidden Memory Access
- NPU memory configuration
- Secure enclave setup
- Event log locations

---

**Status**: Plan Complete - Ready for Implementation
**Priority**: High - Required for full functionality
**Estimated Effort**: 4 weeks full-time development
**Dependencies**: acpidump, iasl, root access
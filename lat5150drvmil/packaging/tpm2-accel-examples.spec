# TPM2 Acceleration Examples - Package Specification
# Package: tpm2-accel-examples_1.0.0-1_all.deb

## Package Metadata

**Name**: tpm2-accel-examples
**Version**: 1.0.0-1
**Architecture**: all (architecture-independent)
**Section**: doc
**Priority**: optional
**Maintainer**: Dell MIL-SPEC Tools Team <milspec@dell.com>
**Installed-Size**: 512 KB

## Dependencies

**Depends**:
- bash (>= 4.4)

**Recommends**:
- tpm2-accel-tools (>= 1.0.0)
- gcc
- make
- python3

**Suggests**:
- tpm2-tools

## Description

Example source code and documentation for using TPM2 acceleration
features on Dell Latitude 5450 MIL-SPEC systems.

This package includes:
 - C examples for all security levels (0-3)
 - Python examples using the bindings
 - Makefiles for compilation
 - Pre-compiled binaries (optional)
 - Workflow documentation
 - Performance benchmarks

Examples demonstrate:
 - Opening /dev/tpm2_accel_early device
 - Using IOCTL interface
 - AES-256-GCM encryption with Intel NPU
 - SHA3-512 hashing
 - Dell military token authorization
 - Security level configuration

## File Manifest

### Examples Directory (/usr/share/doc/tpm2-accel-examples)
```
/usr/share/doc/tpm2-accel-examples/
├── README.md                          # Overview and quick start
├── examples/
│   ├── c/
│   │   ├── 00-basic-status/
│   │   │   ├── basic_status.c
│   │   │   ├── Makefile
│   │   │   └── README.md
│   │   ├── 01-unclassified/
│   │   │   ├── level0_example.c
│   │   │   ├── Makefile
│   │   │   └── README.md
│   │   ├── 02-confidential/
│   │   │   ├── level1_example.c
│   │   │   ├── Makefile
│   │   │   └── README.md
│   │   ├── 03-secret/
│   │   │   ├── secret_level_crypto_example.c
│   │   │   ├── secret_crypto              # Pre-compiled binary
│   │   │   ├── Makefile
│   │   │   └── README.md
│   │   ├── 04-top-secret/
│   │   │   ├── level3_example.c
│   │   │   ├── Makefile
│   │   │   └── README.md
│   │   └── common/
│   │       ├── tpm2_accel_common.h        # Shared header
│   │       └── README.md
│   ├── python/
│   │   ├── 00-basic-status.py
│   │   ├── 01-pcr-translation.py
│   │   ├── 02-me-session.py
│   │   ├── 03-hash-acceleration.py
│   │   ├── 04-full-workflow.py
│   │   └── README.md
│   ├── shell/
│   │   ├── check_acceleration.sh
│   │   ├── test_all_levels.sh
│   │   └── README.md
│   └── Makefile.common                    # Shared makefile rules
├── workflows/
│   ├── SECRET_LEVEL_WORKFLOW.md
│   ├── QUICKSTART_SECRET_LEVEL.md
│   ├── DEVELOPER_GUIDE.md
│   └── API_REFERENCE.md
├── benchmarks/
│   ├── benchmark_results.txt
│   ├── run_benchmarks.sh
│   └── README.md
└── tests/
    ├── test_device_access.sh
    ├── test_ioctl_interface.sh
    └── README.md
```

## Source Files

### 00-basic-status/basic_status.c (NEW)
```c
// Minimal example: Open device and read status
// Security Level: 0 (UNCLASSIFIED)
// Lines: ~100
```

### 01-unclassified/level0_example.c (NEW)
```c
// Full example for security level 0
// Demonstrates: Basic crypto, PCR operations, status queries
// Lines: ~200
```

### 02-confidential/level1_example.c (NEW)
```c
// Full example for security level 1
// Demonstrates: Enhanced crypto, key isolation, secure sessions
// Lines: ~250
```

### 03-secret/secret_level_crypto_example.c (EXISTING)
```c
// Full example for security level 2
// Source: /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/examples/secret_level_crypto_example.c
// Lines: 357
// Status: ✅ Production-ready
```

### 04-top-secret/level3_example.c (NEW)
```c
// Full example for security level 3
// Demonstrates: Maximum security, real-time monitoring, audit trail
// Lines: ~300
```

### common/tpm2_accel_common.h (NEW)
```c
// Shared definitions for all examples
// IOCTL definitions, structures, constants
// Lines: ~150
```

## Python Examples

### 00-basic-status.py (NEW)
```python
#!/usr/bin/env python3
# Minimal Python example: Check device status
# Uses: ctypes with direct IOCTL
# Lines: ~50
```

### 01-pcr-translation.py (NEW)
```python
#!/usr/bin/env python3
# PCR translation example
# Uses: tpm2_accel.bindings (if C library available)
# Lines: ~100
```

### 02-me-session.py (NEW)
```python
#!/usr/bin/env python3
# ME session management example
# Lines: ~150
```

### 03-hash-acceleration.py (NEW)
```python
#!/usr/bin/env python3
# Hardware-accelerated hashing
# Lines: ~120
```

### 04-full-workflow.py (NEW)
```python
#!/usr/bin/env python3
# Complete workflow demonstration
# Lines: ~200
```

## Documentation Files

### README.md (NEW)
```markdown
# TPM2 Acceleration Examples

Quick start guide for using the examples.

## Quick Start

1. Compile C examples:
   cd /usr/share/doc/tpm2-accel-examples/examples/c/03-secret
   make
   sudo ./secret_crypto

2. Run Python examples:
   cd /usr/share/doc/tpm2-accel-examples/examples/python
   python3 00-basic-status.py

## Prerequisites

- TPM2 acceleration kernel module loaded (tpm2_accel_early)
- Device /dev/tpm2_accel_early accessible
- User in 'tpm2-accel' group (for device access)
```

### workflows/DEVELOPER_GUIDE.md (NEW)
```markdown
# Developer Guide - TPM2 Acceleration

## Integrating TPM2 Acceleration into Your Application

### Method 1: Direct IOCTL (No library required)
### Method 2: Using libtpm2-accel.so (C library)
### Method 3: Using Python bindings
```

### workflows/API_REFERENCE.md (NEW)
```markdown
# API Reference - TPM2 Acceleration

Complete reference for:
- IOCTL commands
- Data structures
- Return codes
- Error handling
```

## Installation Scripts

### postinst
```bash
#!/bin/bash
set -e

# Create symlinks in user-accessible location
mkdir -p /usr/local/share/tpm2-accel-examples
ln -sf /usr/share/doc/tpm2-accel-examples/examples /usr/local/share/tpm2-accel-examples/

echo ""
echo "TPM2 acceleration examples installed."
echo ""
echo "Example locations:"
echo "  C examples:      /usr/share/doc/tpm2-accel-examples/examples/c/"
echo "  Python examples: /usr/share/doc/tpm2-accel-examples/examples/python/"
echo "  Workflows:       /usr/share/doc/tpm2-accel-examples/workflows/"
echo ""
echo "Quick start:"
echo "  cd /usr/share/doc/tpm2-accel-examples/examples/c/03-secret"
echo "  make"
echo "  sudo ./secret_crypto"
echo ""
```

### prerm
```bash
#!/bin/bash
set -e

# Remove symlinks
rm -f /usr/local/share/tpm2-accel-examples
rmdir /usr/local/share/tpm2-accel-examples 2>/dev/null || true

exit 0
```

## Build Instructions

### Source Files to Copy

From `/home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/`:
- ✅ `examples/secret_level_crypto_example.c` (existing)
- ✅ `examples/secret_crypto` (compiled binary)
- ✅ `examples/Makefile` (existing)
- ✅ `SECURITY_LEVELS_AND_USAGE.md` (existing)
- ✅ `SECRET_LEVEL_WORKFLOW.md` (existing)
- ✅ `QUICKSTART_SECRET_LEVEL.md` (existing)
- ✅ `check_tpm2_acceleration.sh` (existing)

### Files to Create
- ⚠️ `examples/c/00-basic-status/basic_status.c`
- ⚠️ `examples/c/01-unclassified/level0_example.c`
- ⚠️ `examples/c/02-confidential/level1_example.c`
- ⚠️ `examples/c/04-top-secret/level3_example.c`
- ⚠️ `examples/c/common/tpm2_accel_common.h`
- ⚠️ Python examples (5 files)
- ⚠️ Shell scripts (2 files)
- ⚠️ Documentation (3 files)

### Build Process
```bash
cd /home/john/LAT5150DRVMIL/packaging/
mkdir -p tpm2-accel-examples_1.0.0-1
cd tpm2-accel-examples_1.0.0-1

# Create directory structure
mkdir -p DEBIAN
mkdir -p usr/share/doc/tpm2-accel-examples/{examples/{c/{00-basic-status,01-unclassified,02-confidential,03-secret,04-top-secret,common},python,shell},workflows,benchmarks,tests}

# Copy existing files
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/examples/secret_level_crypto_example.c \
   usr/share/doc/tpm2-accel-examples/examples/c/03-secret/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/examples/secret_crypto \
   usr/share/doc/tpm2-accel-examples/examples/c/03-secret/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/examples/Makefile \
   usr/share/doc/tpm2-accel-examples/examples/c/03-secret/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/SECURITY_LEVELS_AND_USAGE.md \
   usr/share/doc/tpm2-accel-examples/workflows/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/SECRET_LEVEL_WORKFLOW.md \
   usr/share/doc/tpm2-accel-examples/workflows/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/QUICKSTART_SECRET_LEVEL.md \
   usr/share/doc/tpm2-accel-examples/workflows/
cp /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/check_tpm2_acceleration.sh \
   usr/share/doc/tpm2-accel-examples/examples/shell/

# Create control file
cat > DEBIAN/control <<EOF
Package: tpm2-accel-examples
Version: 1.0.0-1
Section: doc
Priority: optional
Architecture: all
Maintainer: Dell MIL-SPEC Tools Team <milspec@dell.com>
Depends: bash (>= 4.4)
Recommends: tpm2-accel-tools (>= 1.0.0), gcc, make, python3
Suggests: tpm2-tools
Installed-Size: 512
Homepage: https://github.com/dell/tpm2-acceleration
Description: Example programs for TPM2 hardware acceleration
 Example source code and documentation for using TPM2 acceleration.
EOF

# Create postinst, prerm scripts
# (see above)

# Set permissions
chmod 755 DEBIAN/postinst DEBIAN/prerm
chmod 755 usr/share/doc/tpm2-accel-examples/examples/c/03-secret/secret_crypto
chmod 644 usr/share/doc/tpm2-accel-examples/examples/c/03-secret/*.c

# Build package
cd ..
dpkg-deb --build tpm2-accel-examples_1.0.0-1
```

### Installation
```bash
sudo dpkg -i tpm2-accel-examples_1.0.0-1_all.deb
```

### Verification
```bash
dpkg -L tpm2-accel-examples
ls -R /usr/share/doc/tpm2-accel-examples/
cd /usr/share/doc/tpm2-accel-examples/examples/c/03-secret
./secret_crypto  # Should run (if module loaded)
```

## Minimal Viable Package (NOW)

**Can be built TODAY** with existing files:

### Included (No new code needed)
- ✅ secret_level_crypto_example.c (357 lines, complete)
- ✅ secret_crypto (compiled binary, works)
- ✅ Makefile (for SECRET level)
- ✅ SECURITY_LEVELS_AND_USAGE.md (631 lines, complete)
- ✅ SECRET_LEVEL_WORKFLOW.md (complete)
- ✅ QUICKSTART_SECRET_LEVEL.md (178 lines, complete)
- ✅ check_tpm2_acceleration.sh (187 lines, complete)

### Excluded (Create later)
- ⚠️ Level 0, 1, 3 examples (can add in v1.1.0)
- ⚠️ Python examples (can add in v1.1.0)
- ⚠️ Advanced documentation (can add in v1.1.0)

### Minimal Package Contents
```
usr/share/doc/tpm2-accel-examples/
├── README.md                              # Brief introduction
├── examples/
│   ├── secret_level_crypto_example.c      # EXISTING
│   ├── secret_crypto                      # EXISTING
│   ├── Makefile                          # EXISTING
│   └── check_tpm2_acceleration.sh        # EXISTING
└── workflows/
    ├── SECURITY_LEVELS_AND_USAGE.md      # EXISTING
    ├── SECRET_LEVEL_WORKFLOW.md          # EXISTING
    └── QUICKSTART_SECRET_LEVEL.md        # EXISTING
```

**Size**: ~50 KB
**Status**: Can build NOW
**Value**: Immediate demonstration of SECRET level capabilities

## Implementation Priority

### Phase 1: Minimal Package (TODAY)
Build package with existing files only. Provides immediate value.

### Phase 2: Complete C Examples (NEXT WEEK)
Add examples for all security levels (0, 1, 3).

### Phase 3: Python Examples (AFTER C LIBRARY)
Add Python examples once libtpm2-accel.so is implemented.

### Phase 4: Advanced Documentation (FUTURE)
Add API reference, developer guide, benchmarks.

## Testing Checklist

- [ ] Install on clean system
- [ ] Verify file locations
- [ ] Compile C example
- [ ] Run compiled binary (with module loaded)
- [ ] Check documentation readability
- [ ] Verify symlinks in /usr/local/share
- [ ] Test removal and purge
- [ ] Check for file conflicts

## Integration Notes

**No conflicts** with existing packages.
**Recommended** by tpm2-accel-tools.
**Independent** of dell-milspec-tools.

## Success Criteria

After installation, users should be able to:
1. ✅ Read documentation in /usr/share/doc
2. ✅ Compile C example
3. ✅ Run example with module loaded
4. ✅ Understand security levels
5. ✅ Use examples as templates for their own code

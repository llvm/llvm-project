# TPM2 Acceleration Tools - Package Specification
# Package: tpm2-accel-tools_1.0.0-1_amd64.deb

## Package Metadata

**Name**: tpm2-accel-tools
**Version**: 1.0.0-1
**Architecture**: amd64
**Section**: utils
**Priority**: optional
**Maintainer**: Dell MIL-SPEC Tools Team <milspec@dell.com>
**Installed-Size**: 2048 KB

## Dependencies

**Depends**:
- bash (>= 4.4)
- coreutils
- tpm2-accel-early-dkms (>= 1.0.0)

**Recommends**:
- tpm2-tools
- dell-milspec-tools

**Suggests**:
- tpm2-accel-examples

## Description

Command-line utilities and libraries for interacting with the
TPM2 acceleration kernel module (tpm2_accel_early).

This package provides:
 - Configuration tools (tpm2-accel-configure)
 - Status monitoring (tpm2-accel-status)
 - Testing utilities (tpm2-accel-test)
 - Benchmark tools (tpm2-accel-benchmark)
 - C library for application development (libtpm2-accel)
 - Python bindings for scripting

Supports 4 security levels (UNCLASSIFIED to TOP SECRET) with
Intel NPU/GNA/ME hardware acceleration on Dell Latitude 5450 MIL-SPEC.

## File Manifest

### Executables (/usr/bin)
```
/usr/bin/tpm2-accel-status         # 159 lines, status query
/usr/bin/tpm2-accel-configure      # TBD, configuration tool
/usr/bin/tpm2-accel-test           # TBD, test suite
/usr/bin/tpm2-accel-benchmark      # TBD, performance tool
```

### Libraries (/usr/lib/x86_64-linux-gnu)
```
/usr/lib/x86_64-linux-gnu/libtpm2-accel.so.1.0.0    # Shared library
/usr/lib/x86_64-linux-gnu/libtpm2-accel.so.1        # Symlink
/usr/lib/x86_64-linux-gnu/libtpm2-accel.so          # Dev symlink
```

### Python Module (/usr/lib/python3/dist-packages)
```
/usr/lib/python3/dist-packages/tpm2_accel/__init__.py
/usr/lib/python3/dist-packages/tpm2_accel/bindings.py
/usr/lib/python3/dist-packages/tpm2_accel/py.typed
```

### Development Headers (/usr/include)
```
/usr/include/tpm2-accel/tpm2_compat_accelerated.h
```

### pkg-config (/usr/lib/pkgconfig)
```
/usr/lib/pkgconfig/tpm2-accel.pc
```

### Documentation (/usr/share/doc/tpm2-accel-tools)
```
/usr/share/doc/tpm2-accel-tools/SECURITY_LEVELS_AND_USAGE.md
/usr/share/doc/tpm2-accel-tools/QUICKSTART_SECRET_LEVEL.md
/usr/share/doc/tpm2-accel-tools/INSTALLATION_GUIDE.md
/usr/share/doc/tpm2-accel-tools/README.md
/usr/share/doc/tpm2-accel-tools/changelog.gz
/usr/share/doc/tpm2-accel-tools/copyright
```

### Man Pages (/usr/share/man)
```
/usr/share/man/man1/tpm2-accel-status.1.gz
/usr/share/man/man1/tpm2-accel-test.1.gz
/usr/share/man/man1/tpm2-accel-benchmark.1.gz
/usr/share/man/man8/tpm2-accel-configure.8.gz
```

## Installation Scripts

### postinst
```bash
#!/bin/bash
set -e

# Create tpm2-accel group
if ! getent group tpm2-accel >/dev/null; then
    groupadd -r tpm2-accel
fi

# Set device permissions if module loaded
if [ -c /dev/tpm2_accel_early ]; then
    chgrp tpm2-accel /dev/tpm2_accel_early
    chmod 0660 /dev/tpm2_accel_early
fi

# Create udev rule
cat > /etc/udev/rules.d/99-tpm2-accel.rules <<'EOF'
KERNEL=="tpm2_accel_early", GROUP="tpm2-accel", MODE="0660"
EOF

udevadm control --reload-rules

# Update library cache
ldconfig

echo "TPM2 acceleration tools installed successfully."
echo "Add users to 'tpm2-accel' group: sudo usermod -a -G tpm2-accel USERNAME"
```

### prerm
```bash
#!/bin/bash
set -e

# Nothing to do on removal
exit 0
```

### postrm
```bash
#!/bin/bash
set -e

case "$1" in
    purge)
        # Remove group on purge
        if getent group tpm2-accel >/dev/null; then
            groupdel tpm2-accel || true
        fi

        # Remove udev rule
        rm -f /etc/udev/rules.d/99-tpm2-accel.rules
        udevadm control --reload-rules || true
        ;;
    remove|upgrade|failed-upgrade|abort-install|abort-upgrade|disappear)
        # Update library cache
        ldconfig
        ;;
esac

exit 0
```

## Build Instructions

### Prerequisites
```bash
apt-get install build-essential debhelper dh-make devscripts
```

### Build Process
```bash
cd /home/john/LAT5150DRVMIL/packaging/
mkdir -p tpm2-accel-tools_1.0.0-1
cd tpm2-accel-tools_1.0.0-1

# Create directory structure
mkdir -p DEBIAN
mkdir -p usr/{bin,lib/x86_64-linux-gnu,lib/python3/dist-packages/tpm2_accel,include/tpm2-accel,lib/pkgconfig}
mkdir -p usr/share/{doc/tpm2-accel-tools,man/man1,man/man8}

# Copy files
cp /home/john/LAT5150DRVMIL/packaging/dell-milspec-tools/usr/bin/tpm2-accel-status usr/bin/
# Add other tools when ready

# Create control file
cat > DEBIAN/control <<EOF
Package: tpm2-accel-tools
Version: 1.0.0-1
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Dell MIL-SPEC Tools Team <milspec@dell.com>
Depends: bash (>= 4.4), coreutils, tpm2-accel-early-dkms (>= 1.0.0)
Recommends: tpm2-tools, dell-milspec-tools
Suggests: tpm2-accel-examples
Installed-Size: 2048
Homepage: https://github.com/dell/tpm2-acceleration
Description: Userspace tools for TPM2 hardware acceleration
 Command-line utilities and libraries for TPM2 acceleration.
EOF

# Create postinst, prerm, postrm scripts
# (see above)

# Set permissions
chmod 755 DEBIAN/postinst DEBIAN/prerm DEBIAN/postrm
chmod 755 usr/bin/*

# Build package
cd ..
dpkg-deb --build tpm2-accel-tools_1.0.0-1
```

### Installation
```bash
sudo dpkg -i tpm2-accel-tools_1.0.0-1_amd64.deb
```

### Verification
```bash
dpkg -L tpm2-accel-tools          # List files
tpm2-accel-status                 # Run status check
dpkg -s tpm2-accel-tools          # Show package info
```

## Implementation Status

### Completed
- ✅ Package specification
- ✅ tpm2-accel-status (existing)
- ✅ File layout design
- ✅ Installation scripts

### Pending Implementation
- ⚠️ C library (libtpm2-accel.so) - headers exist, implementation needed
- ⚠️ tpm2-accel-configure - needs writing
- ⚠️ tpm2-accel-test - needs writing
- ⚠️ tpm2-accel-benchmark - needs writing
- ⚠️ Python packaging - code exists, packaging needed
- ⚠️ Man pages - need writing

### Blockers
- **C Library**: Python bindings depend on this
- **Tools**: Can be implemented with direct IOCTL, don't require C library

## Testing Checklist

- [ ] Install on clean Debian system
- [ ] Verify module dependency (tpm2-accel-early-dkms)
- [ ] Check /dev/tpm2_accel_early permissions
- [ ] Verify tpm2-accel group creation
- [ ] Test tpm2-accel-status command
- [ ] Check udev rule creation
- [ ] Verify library in ldconfig cache
- [ ] Test removal and purge
- [ ] Check for file conflicts with dell-milspec-tools

## Integration Notes

### Conflicts
**None** - tpm2-accel-status in both dell-milspec-tools and tpm2-accel-tools

**Resolution**: Keep in dell-milspec-tools, remove from tpm2-accel-tools

### Alternatives
Use Debian alternatives system:
```bash
update-alternatives --install /usr/bin/tpm2-accel-status tpm2-accel-status \
    /usr/bin/tpm2-accel-status-v1 100
```

### Recommendation
Keep tpm2-accel-status ONLY in dell-milspec-tools package to avoid conflicts.

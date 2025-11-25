# TPM2 Early Boot Acceleration - Debian Package Index

## Quick Navigation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[README.md](README.md)** - Complete documentation (12 KB)
- **[PACKAGE_MANIFEST.md](PACKAGE_MANIFEST.md)** - Detailed file manifest (8 KB)
- **[FILES_SUMMARY.txt](FILES_SUMMARY.txt)** - Summary reference

## Build and Test

```bash
# Build package
./build-package.sh

# Test package
./test-package.sh

# Install package
sudo dpkg -i build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb
```

## Package Information

- **Name**: tpm2-accel-early-dkms
- **Version**: 1.0.0-1
- **Architecture**: amd64
- **Size**: ~25-30 KB (compressed)
- **License**: GPL-2.0
- **Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

## Directory Structure

```
packaging/debian/
├── DEBIAN/                           # Package control files
│   ├── control                       # Metadata (1.3 KB)
│   ├── postinst                      # Post-install script (4.4 KB, executable)
│   ├── prerm                         # Pre-removal script (2.1 KB, executable)
│   ├── postrm                        # Post-removal script (2.4 KB, executable)
│   ├── copyright                     # GPL-2.0 license (2.0 KB)
│   ├── changelog                     # Version history (1.9 KB)
│   ├── compat                        # Debhelper v14 (3 bytes)
│   ├── conffiles                     # Config file list (82 bytes)
│   └── triggers                      # Initramfs triggers (103 bytes)
│
├── lib/systemd/system/               # Systemd service
│   └── tpm2-acceleration-early.service  # Early boot service
│
├── build-package.sh                  # Build script (executable)
├── test-package.sh                   # Test script (executable)
│
├── INDEX.md                          # This file
├── README.md                         # Full documentation
├── QUICKSTART.md                     # Quick start guide
├── PACKAGE_MANIFEST.md               # Complete file manifest
└── FILES_SUMMARY.txt                 # Summary reference
```

## Complete File List (22 files)

### DEBIAN Control Files (9 files)
1. `DEBIAN/control` - Package metadata and dependencies
2. `DEBIAN/postinst` - Post-installation script (DKMS build/install)
3. `DEBIAN/prerm` - Pre-removal script (module unload)
4. `DEBIAN/postrm` - Post-removal cleanup and purge
5. `DEBIAN/copyright` - GPL-2.0 license information
6. `DEBIAN/changelog` - Version history
7. `DEBIAN/compat` - Debhelper compatibility (14)
8. `DEBIAN/conffiles` - Configuration files marker
9. `DEBIAN/triggers` - Initramfs update triggers

### Systemd Service (1 file)
10. `lib/systemd/system/tpm2-acceleration-early.service` - Early boot service

### Scripts (2 files)
11. `build-package.sh` - Package build automation
12. `test-package.sh` - Package validation (10 tests)

### Documentation (5 files)
13. `INDEX.md` - This navigation file
14. `README.md` - Comprehensive documentation
15. `QUICKSTART.md` - 5-minute quick start
16. `PACKAGE_MANIFEST.md` - Detailed file breakdown
17. `FILES_SUMMARY.txt` - Quick reference summary

### Source Files (4 files, copied during build)
18. `tpm2_accel_early.c` - Kernel module source (~1,180 lines)
19. `tpm2_accel_early.h` - Kernel module header
20. `Makefile` - Kernel build configuration
21. `dkms.conf` - DKMS build configuration

### Runtime Config (2 files, created at install)
22. `/etc/modprobe.d/tpm2-acceleration.conf` - Module parameters
23. `/etc/modules-load.d/tpm2-acceleration.conf` - Early load config

## Key Features

### DKMS Integration
- Automatic kernel module building for all installed kernels
- Automatic rebuilds on kernel updates
- Clean removal on package uninstall

### Early Boot Support
- Module loads during system initialization
- Initramfs integration for pre-boot availability
- Systemd service for early startup

### Hardware Acceleration
- Intel NPU (34.0 TOPS)
- Intel GNA 3.5 security monitoring
- Intel Management Engine integration
- TPM 2.0 hardware acceleration

### Security Features
- Multi-level classification (0-3: UNCLASSIFIED → TOP SECRET)
- Dell SMBIOS military token authorization (0x049e-0x04a3)
- Hardware-level access control
- Secure boot compatible

### Configuration
- Default security level: UNCLASSIFIED (0)
- Configurable via `/etc/modprobe.d/tpm2-acceleration.conf`
- Persistent across reboots
- Initramfs integration for early boot security

## Installation Workflow

1. **Build** → `./build-package.sh`
2. **Test** → `./test-package.sh` (optional)
3. **Install** → `sudo dpkg -i build/tpm2-accel-early-dkms_*.deb`
4. **Verify** → `dkms status tpm2-accel-early`
5. **Configure** → Edit `/etc/modprobe.d/tpm2-acceleration.conf` (optional)
6. **Reboot** → Module loads automatically

## Verification Commands

```bash
# Check DKMS status
dkms status tpm2-accel-early

# Verify module loaded
lsmod | grep tpm2_accel_early

# Check device node
ls -l /dev/tpm2_accel_early

# View kernel messages
dmesg | grep tpm2_accel_early

# Check systemd service
systemctl status tpm2-acceleration-early.service

# View module information
modinfo tpm2_accel_early

# Check configuration
cat /etc/modprobe.d/tpm2-acceleration.conf
```

## Troubleshooting

### Module Won't Load
```bash
# Check TPM hardware
ls -l /dev/tpm*

# Rebuild module
sudo dkms build -m tpm2-accel-early -v 1.0.0
sudo dkms install -m tpm2-accel-early -v 1.0.0

# Check build log
cat /var/lib/dkms/tpm2-accel-early/1.0.0/build/make.log
```

### Package Build Issues
```bash
# Check prerequisites
which dpkg-deb
which dkms

# Verify source files exist
ls -l ../../tpm2_compat/c_acceleration/kernel_module/
ls -l ../dkms/tpm2-accel-early.dkms.conf
```

### Service Won't Start
```bash
# Check service status
systemctl status tpm2-acceleration-early.service

# View logs
journalctl -u tpm2-acceleration-early.service

# Reload systemd
sudo systemctl daemon-reload
```

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| Processor | Intel Core Ultra 7 165H (Meteor Lake) |
| TPM | TPM 2.0 hardware module |
| Platform | Dell Latitude 5450 MIL-SPEC |
| Kernel | Linux 6.14.0 or newer |
| RAM | Minimum 4GB |
| Storage | Minimum 50MB for DKMS builds |

## Dependencies

### Build Time
- build-essential
- dpkg-dev
- debhelper (>= 14)

### Runtime
- dkms (>= 2.1.0.0)
- linux-headers-generic or linux-headers-$(uname -r)

### Recommended
- tpm2-tools
- systemd

## Security Levels

| Level | Classification | Use Case |
|-------|---------------|----------|
| 0 | UNCLASSIFIED | Default, general purpose |
| 1 | CONFIDENTIAL | Sensitive but unclassified |
| 2 | SECRET | Classified operations |
| 3 | TOP SECRET | Highest classification |

**To change security level:**
1. Edit `/etc/modprobe.d/tpm2-acceleration.conf`
2. Change `security_level=0` to desired level (0-3)
3. Run `sudo update-initramfs -u`
4. Reboot system

## Package Contents Summary

| Category | Count | Total Size |
|----------|-------|------------|
| DEBIAN control files | 9 | ~15 KB |
| Systemd service | 1 | ~1 KB |
| Scripts | 2 | ~14 KB |
| Documentation | 5 | ~25 KB |
| Source files (in package) | 4 | ~55 KB |
| **Total** | **21** | **~110 KB** |
| **Compressed package** | **1** | **~25-30 KB** |

## Support and Contact

- **Project**: Military TPM2 Acceleration Project
- **License**: GPL-2.0
- **Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
- **Documentation**: See README.md for complete details

## Quick Command Reference

```bash
# Build
./build-package.sh

# Test
./test-package.sh

# Install
sudo dpkg -i build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb

# Status
dkms status

# Verify
lsmod | grep tpm2_accel_early

# Configure
sudo nano /etc/modprobe.d/tpm2-acceleration.conf
sudo update-initramfs -u
sudo reboot

# Remove
sudo apt-get remove tpm2-accel-early-dkms

# Purge
sudo apt-get purge tpm2-accel-early-dkms
```

## Next Steps

1. **First time?** → Read [QUICKSTART.md](QUICKSTART.md)
2. **Need details?** → Read [README.md](README.md)
3. **Want file info?** → Read [PACKAGE_MANIFEST.md](PACKAGE_MANIFEST.md)
4. **Ready to build?** → Run `./build-package.sh`
5. **Need to verify?** → Run `./test-package.sh`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Package Version**: 1.0.0-1

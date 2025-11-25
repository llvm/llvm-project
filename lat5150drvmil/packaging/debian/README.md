# TPM2 Early Boot Acceleration - Debian Package

Complete Debian/DKMS packaging for the `tpm2-accel-early` kernel module.

## Package Information

- **Package Name**: tpm2-accel-early-dkms
- **Version**: 1.0.0-1
- **Architecture**: amd64
- **License**: GPL-2.0
- **Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

## Features

- **DKMS Integration**: Automatic kernel module building for all installed kernels
- **Early Boot Support**: Module loads during system initialization via systemd
- **Initramfs Integration**: Module available during pre-boot phase
- **Hardware Acceleration**: Intel NPU (34.0 TOPS), GNA 3.5, Management Engine
- **Security Levels**: 0-3 (UNCLASSIFIED → TOP SECRET)
- **Dell SMBIOS**: Military token integration (0x049e-0x04a3)

## Directory Structure

```
debian/
├── DEBIAN/
│   ├── control          # Package metadata and dependencies
│   ├── postinst         # Post-installation script (DKMS build/install)
│   ├── prerm            # Pre-removal script (unload module)
│   ├── postrm           # Post-removal script (cleanup)
│   ├── copyright        # GPL-2.0 license information
│   ├── changelog        # Package version history
│   ├── compat           # Debhelper compatibility level (14)
│   ├── conffiles        # Configuration files list
│   └── triggers         # Initramfs update triggers
├── lib/systemd/system/
│   └── tpm2-acceleration-early.service  # Systemd service unit
├── build-package.sh     # Build script
├── test-package.sh      # Testing script
└── README.md           # This file
```

## Building the Package

### Prerequisites

```bash
sudo apt-get install -y \
    build-essential \
    dkms \
    debhelper \
    dpkg-dev \
    linux-headers-$(uname -r)
```

### Build Process

```bash
# Navigate to packaging directory
cd /home/john/LAT5150DRVMIL/packaging/debian

# Run build script
./build-package.sh
```

The built package will be located at:
```
build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb
```

## Installation

### From .deb Package

```bash
# Install the package
sudo dpkg -i build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb

# Install any missing dependencies
sudo apt-get install -f
```

### Post-Installation Verification

```bash
# Check DKMS status
dkms status tpm2-accel-early

# Verify module is loaded
lsmod | grep tpm2_accel_early

# Check device node
ls -l /dev/tpm2_accel_early

# View kernel messages
dmesg | grep tpm2_accel_early

# Check systemd service
systemctl status tpm2-acceleration-early.service
```

## Configuration

### Module Parameters

The module accepts three parameters (configured in `/etc/modprobe.d/tpm2-acceleration.conf`):

1. **security_level**: Security classification level
   - `0` = UNCLASSIFIED (default)
   - `1` = CONFIDENTIAL
   - `2` = SECRET
   - `3` = TOP SECRET

2. **early_init**: Early boot initialization
   - `1` = Enabled (default)
   - `0` = Disabled

3. **debug_mode**: Debug logging
   - `0` = Disabled (default)
   - `1` = Enabled

### Changing Configuration

```bash
# Edit configuration
sudo nano /etc/modprobe.d/tpm2-acceleration.conf

# Example: Enable SECRET level
# options tpm2_accel_early early_init=1 debug_mode=0 security_level=2

# Update initramfs
sudo update-initramfs -u

# Reboot to apply changes
sudo reboot
```

### Disabling at Boot

To prevent the module from loading, add to kernel command line:
```
tpm2_accel_early.disable
```

## File Locations

### Installed Files

- **Module Source**: `/usr/src/tpm2-accel-early-1.0.0/`
- **Built Module**: `/lib/modules/$(uname -r)/updates/dkms/tpm2_accel_early.ko`
- **Device Node**: `/dev/tpm2_accel_early`
- **Systemd Service**: `/lib/systemd/system/tpm2-acceleration-early.service`

### Configuration Files

- **Modprobe Config**: `/etc/modprobe.d/tpm2-acceleration.conf`
- **Modules Load**: `/etc/modules-load.d/tpm2-acceleration.conf`

### Documentation

- **Package Docs**: `/usr/share/doc/tpm2-accel-early-dkms/`
- **README**: `/usr/share/doc/tpm2-accel-early-dkms/README.Debian`
- **Changelog**: `/usr/share/doc/tpm2-accel-early-dkms/changelog.Debian.gz`
- **Copyright**: `/usr/share/doc/tpm2-accel-early-dkms/copyright`

## Removal

### Remove Package (Keep Configuration)

```bash
sudo apt-get remove tpm2-accel-early-dkms
```

### Purge Package (Remove Everything)

```bash
sudo apt-get purge tpm2-accel-early-dkms
# or
sudo dpkg --purge tpm2-accel-early-dkms
```

## Hardware Requirements

- **Processor**: Intel Core Ultra 7 165H (Meteor Lake) or compatible
- **TPM**: TPM 2.0 hardware module
- **Platform**: Dell Latitude 5450 MIL-SPEC or compatible
- **Kernel**: Linux 6.14.0 or newer

## DKMS Workflow

### Automatic Rebuilds

DKMS automatically rebuilds the module when:
- New kernel is installed
- Kernel headers are updated
- Package is reinstalled

### Manual Operations

```bash
# Check status
dkms status

# Build for specific kernel
dkms build -m tpm2-accel-early -v 1.0.0 -k $(uname -r)

# Install for specific kernel
dkms install -m tpm2-accel-early -v 1.0.0 -k $(uname -r)

# Rebuild for all kernels
dkms autoinstall -m tpm2-accel-early -v 1.0.0

# Remove from specific kernel
dkms uninstall -m tpm2-accel-early -v 1.0.0 -k $(uname -r)
```

## Troubleshooting

### Module Won't Load

```bash
# Check DKMS build log
cat /var/lib/dkms/tpm2-accel-early/1.0.0/build/make.log

# Check kernel compatibility
uname -r
dkms status tpm2-accel-early

# Rebuild module
sudo dkms remove -m tpm2-accel-early -v 1.0.0 --all
sudo dkms add -m tpm2-accel-early -v 1.0.0
sudo dkms build -m tpm2-accel-early -v 1.0.0
sudo dkms install -m tpm2-accel-early -v 1.0.0
```

### TPM Not Detected

```bash
# Check TPM device
ls -l /dev/tpm*

# Check kernel modules
lsmod | grep tpm

# Load TPM modules
sudo modprobe tpm_tis
sudo modprobe tpm_crb
```

### Device Node Missing

```bash
# Check if module loaded
lsmod | grep tpm2_accel_early

# Check kernel messages
dmesg | tail -50

# Try loading manually
sudo modprobe tpm2_accel_early early_init=1 security_level=0
```

### Service Won't Start

```bash
# Check service status
systemctl status tpm2-acceleration-early.service

# View service logs
journalctl -u tpm2-acceleration-early.service

# Reload systemd
sudo systemctl daemon-reload
sudo systemctl restart tpm2-acceleration-early.service
```

## Security Considerations

### Dell SMBIOS Tokens

The module requires Dell SMBIOS military tokens (0x049e-0x04a3) for hardware authorization. These tokens are:

- **Platform-Specific**: Only available on Dell MIL-SPEC systems
- **Hardware-Enforced**: Validated by Dell firmware
- **Security-Level Dependent**: Different tokens for different classification levels

### Security Levels

| Level | Classification | Use Case |
|-------|---------------|----------|
| 0 | UNCLASSIFIED | General purpose, development |
| 1 | CONFIDENTIAL | Sensitive but unclassified |
| 2 | SECRET | Classified operations |
| 3 | TOP SECRET | Highest classification |

**WARNING**: Deploying at levels 2-3 requires appropriate security clearances and compliance with organizational security policies.

## Development

### Modifying the Module

1. Update source files in `/usr/src/tpm2-accel-early-1.0.0/`
2. Rebuild with DKMS: `dkms build -m tpm2-accel-early -v 1.0.0`
3. Reinstall: `dkms install -m tpm2-accel-early -v 1.0.0`

### Creating Updated Package

1. Update source files in `../../tpm2_compat/c_acceleration/kernel_module/`
2. Update `DEBIAN/changelog` with new version
3. Update `DEBIAN/control` version
4. Run `./build-package.sh`

## Testing

Run the included test script:

```bash
./test-package.sh
```

This will:
- Verify package integrity
- Check dependencies
- Validate file structure
- Test installation/removal

## Support

For issues, questions, or contributions:

- **Project**: Military TPM2 Acceleration Project
- **Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
- **License**: GPL-2.0

## References

- DKMS Documentation: `/usr/share/doc/dkms/`
- Debian Policy: https://www.debian.org/doc/debian-policy/
- TPM 2.0 Specifications: https://trustedcomputinggroup.org/
- Intel NPU Documentation: Intel Developer Zone

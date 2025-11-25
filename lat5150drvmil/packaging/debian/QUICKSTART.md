# Quick Start Guide - TPM2 Early Boot Acceleration DKMS Package

## 1. Build Package (30 seconds)

```bash
cd /home/john/LAT5150DRVMIL/packaging/debian
./build-package.sh
```

**Output**: `build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb`

## 2. Test Package (Optional)

```bash
./test-package.sh
```

## 3. Install Package

```bash
sudo dpkg -i build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb
sudo apt-get install -f  # if dependencies missing
```

## 4. Verify Installation

```bash
# Check DKMS built the module
dkms status tpm2-accel-early

# Check module loaded
lsmod | grep tpm2_accel_early

# Check device node
ls -l /dev/tpm2_accel_early

# View kernel messages
dmesg | grep tpm2_accel_early

# Check systemd service
systemctl status tpm2-acceleration-early.service
```

## 5. Configure (Optional)

### Change Security Level

```bash
# Edit configuration
sudo nano /etc/modprobe.d/tpm2-acceleration.conf

# Change security_level parameter:
# 0 = UNCLASSIFIED (default)
# 1 = CONFIDENTIAL
# 2 = SECRET
# 3 = TOP SECRET

# Apply changes
sudo update-initramfs -u
sudo reboot
```

### Enable Debug Mode

```bash
# Edit configuration
sudo nano /etc/modprobe.d/tpm2-acceleration.conf

# Change debug_mode=0 to debug_mode=1

# Apply changes
sudo update-initramfs -u
sudo reboot
```

## 6. Troubleshooting

### Module Won't Load

```bash
# Check TPM hardware
ls -l /dev/tpm*

# Check kernel logs
journalctl -k | grep tpm

# Rebuild module
sudo dkms build -m tpm2-accel-early -v 1.0.0
sudo dkms install -m tpm2-accel-early -v 1.0.0
```

### View Detailed Logs

```bash
# Kernel messages
dmesg | grep tpm2_accel_early

# Service logs
journalctl -u tpm2-acceleration-early.service

# DKMS build log
cat /var/lib/dkms/tpm2-accel-early/1.0.0/build/make.log
```

## 7. Uninstall

```bash
# Remove package (keep config)
sudo apt-get remove tpm2-accel-early-dkms

# Purge everything
sudo apt-get purge tpm2-accel-early-dkms
```

## Common Commands Reference

| Task | Command |
|------|---------|
| Build package | `./build-package.sh` |
| Test package | `./test-package.sh` |
| Install | `sudo dpkg -i build/tpm2-accel-early-dkms_*.deb` |
| Check status | `dkms status tpm2-accel-early` |
| View module info | `modinfo tpm2_accel_early` |
| Load manually | `sudo modprobe tpm2_accel_early` |
| Unload manually | `sudo modprobe -r tpm2_accel_early` |
| Update initramfs | `sudo update-initramfs -u` |
| Service status | `systemctl status tpm2-acceleration-early.service` |

## Files Installed

- **Module**: `/lib/modules/$(uname -r)/updates/dkms/tpm2_accel_early.ko`
- **Device**: `/dev/tpm2_accel_early`
- **Config**: `/etc/modprobe.d/tpm2-acceleration.conf`
- **Service**: `/lib/systemd/system/tpm2-acceleration-early.service`

## Hardware Requirements

- Intel Core Ultra 7 165H (Meteor Lake)
- TPM 2.0 hardware
- Dell Latitude 5450 MIL-SPEC
- Kernel 6.14.0+

## Support

For issues: Check `/usr/share/doc/tpm2-accel-early-dkms/README.Debian`

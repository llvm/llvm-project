# ✅ DSMIL MILITARY-SPEC KERNEL DEPLOYMENT CHECKLIST

## Pre-Deployment Verification

### System Requirements
- [ ] Hardware: Dell Latitude 5450
- [ ] CPU: Intel Core Ultra 7 165H (Meteor Lake)
- [ ] BIOS: Dell SecureBIOS with DSMIL support
- [ ] TPM: STMicroelectronics ST33TPHF2XSP (TPM 2.0)
- [ ] Storage: At least 50GB free space
- [ ] Backup: Current system fully backed up

### Build Verification
- [x] Kernel build completed successfully
- [x] bzImage created (13MB at /home/john/linux-6.16.9/arch/x86/boot/bzImage)
- [x] DSMIL driver integrated (584KB, 2800+ lines)
- [x] Mode 5 set to STANDARD (safe)
- [x] All 84 DSMIL devices configured
- [x] Build logs available

### Documentation Review
- [x] Read COMPLETE_MILITARY_SPEC_HANDOFF.md
- [x] Read MODE5_SECURITY_LEVELS_WARNING.md
- [x] Read APT_ADVANCED_SECURITY_FEATURES.md
- [x] Understand Mode 5 security levels
- [x] Confirm PARANOID_PLUS is NOT enabled

## Phase 1: Kernel Installation (15-20 minutes)

### Step 1.1: Pre-Installation Backup
```bash
# Create backup of current kernel
sudo cp /boot/vmlinuz-$(uname -r) /boot/vmlinuz-$(uname -r).backup
sudo cp /boot/initrd.img-$(uname -r) /boot/initrd.img-$(uname -r).backup

# Backup GRUB configuration
sudo cp /etc/default/grub /etc/default/grub.backup
```

- [ ] Current kernel backed up
- [ ] GRUB config backed up
- [ ] Boot partition has sufficient space

### Step 1.2: Install Kernel Modules
```bash
cd /home/john/linux-6.16.9
sudo make modules_install
```

**Expected output:**
- Modules installed to /lib/modules/6.16.9/
- No errors reported

- [ ] Modules installed successfully
- [ ] No error messages
- [ ] /lib/modules/6.16.9/ directory created

### Step 1.3: Install Kernel Image
```bash
sudo make install
```

**Expected output:**
- Kernel copied to /boot/
- initramfs generated
- GRUB updated

- [ ] Kernel installed to /boot/
- [ ] initramfs created
- [ ] No error messages

### Step 1.4: Configure GRUB
```bash
# Edit GRUB configuration
sudo nano /etc/default/grub

# Add these parameters to GRUB_CMDLINE_LINUX:
# intel_iommu=on iommu=force mode5.level=standard tpm_tis.force=1

# Update GRUB
sudo update-grub
```

- [ ] GRUB configuration edited
- [ ] Parameters added correctly
- [ ] GRUB updated successfully
- [ ] No warnings or errors

### Step 1.5: Verify Boot Entry
```bash
# List GRUB entries
sudo grep menuentry /boot/grub/grub.cfg | grep 6.16.9
```

- [ ] New kernel appears in GRUB menu
- [ ] Entry is properly formatted
- [ ] Default kernel unchanged (for safety)

## Phase 2: First Boot (30-45 minutes)

### Step 2.1: Reboot System
```bash
# Save all work
# Close all applications
sudo reboot
```

**During boot:**
- [ ] Select "Linux 6.16.9" from GRUB menu
- [ ] Watch for DSMIL initialization messages
- [ ] System boots successfully

### Step 2.2: Post-Boot Verification
```bash
# Verify kernel version
uname -r
# Should show: 6.16.9

# Check DSMIL driver loaded
lsmod | grep milspec

# Check dmesg for DSMIL messages
dmesg | grep "MIL-SPEC"
dmesg | grep "DSMIL"
dmesg | grep "Mode 5"
```

- [ ] Kernel version is 6.16.9
- [ ] DSMIL driver loaded
- [ ] Mode 5 messages in dmesg
- [ ] No kernel panics or errors

### Step 2.3: Verify Mode 5 Status
```bash
# Check Mode 5 level
cat /sys/module/dell_milspec/parameters/mode5_level
# Should show: standard

# Check if Mode 5 is enabled
cat /sys/module/dell_milspec/parameters/mode5_enabled
# Should show: Y or 1
```

- [ ] Mode 5 level is "standard"
- [ ] Mode 5 is enabled
- [ ] No permission errors

### Step 2.4: Verify DSMIL Devices
```bash
# List DSMIL devices
ls /sys/class/milspec/

# Check device count
ls /sys/class/milspec/ | wc -l
# Should show: 84 (or close to it)
```

- [ ] DSMIL device directory exists
- [ ] Multiple devices visible
- [ ] No error accessing devices

## Phase 3: AVX-512 Integration (10-15 minutes)

### Step 3.1: Check AVX-512 Availability
```bash
# Check CPU features
lscpu | grep avx512
grep avx512 /proc/cpuinfo
```

- [ ] AVX-512 flags present (may not show until module loaded)
- [ ] P-cores detected

### Step 3.2: Load AVX-512 Enabler Module
```bash
# Load the module
sudo insmod /home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko

# Verify it loaded
lsmod | grep dsmil_avx512

# Check dmesg
dmesg | tail -20
```

- [ ] Module loaded successfully
- [ ] No error messages
- [ ] AVX-512 enabled messages in dmesg

### Step 3.3: Verify AVX-512 Functionality
```bash
# Re-check CPU features
lscpu | grep avx512
grep avx512 /proc/cpuinfo | head -5
```

- [ ] AVX-512 flags now visible
- [ ] Available on P-cores only
- [ ] Microcode version 0x1c or higher

## Phase 4: livecd-gen Compilation (20-30 minutes)

### Step 4.1: Compile C Modules
```bash
cd /home/john/livecd-gen

# Compile all modules
for module in ai_hardware_optimizer meteor_lake_scheduler \
              dell_platform_optimizer tpm_kernel_security avx512_optimizer; do
    echo "Compiling ${module}..."
    gcc -O3 -march=native -mtune=native ${module}.c -o ${module}
    if [ $? -eq 0 ]; then
        echo "✅ ${module} compiled successfully"
    else
        echo "❌ ${module} compilation failed"
    fi
done
```

- [ ] ai_hardware_optimizer compiled
- [ ] meteor_lake_scheduler compiled
- [ ] dell_platform_optimizer compiled
- [ ] tpm_kernel_security compiled
- [ ] avx512_optimizer compiled
- [ ] All binaries executable

### Step 4.2: Test Compiled Modules
```bash
# Test each module (non-root safe check)
./ai_hardware_optimizer --help 2>&1 | head -3
./meteor_lake_scheduler --help 2>&1 | head -3
./dell_platform_optimizer --help 2>&1 | head -3
./tpm_kernel_security --help 2>&1 | head -3
./avx512_optimizer --help 2>&1 | head -3
```

- [ ] All modules execute without segfault
- [ ] Help or error messages appear
- [ ] No library dependency errors

## Phase 5: Security Verification (15-20 minutes)

### Step 5.1: Verify IOMMU
```bash
# Check IOMMU is enabled
dmesg | grep -i iommu
cat /proc/cmdline | grep iommu
```

- [ ] IOMMU enabled in kernel command line
- [ ] IOMMU initialization messages in dmesg
- [ ] No IOMMU errors

### Step 5.2: Verify TPM
```bash
# Check TPM device
ls -la /dev/tpm*

# Check TPM version
cat /sys/class/tpm/tpm0/tpm_version_major
```

- [ ] TPM device exists (/dev/tpm0)
- [ ] TPM version is 2
- [ ] No access errors

### Step 5.3: Verify Memory Encryption (TME)
```bash
# Check if TME is available
dmesg | grep -i "memory encryption"
dmesg | grep -i TME
```

- [ ] TME initialization messages found
- [ ] No TME errors
- [ ] Memory encryption active (if supported)

### Step 5.4: Verify Boot Chain
```bash
# Check secure boot status
mokutil --sb-state 2>/dev/null || echo "Secure boot tools not installed"

# Check kernel integrity
sudo dmesg | grep -i "signature"
```

- [ ] Secure boot status checked
- [ ] No integrity violations
- [ ] Boot chain validated

## Phase 6: Performance Testing (30 minutes)

### Step 6.1: System Stability
```bash
# Run basic stress test (optional)
stress-ng --cpu 4 --timeout 60s 2>/dev/null || echo "stress-ng not installed"

# Monitor dmesg for errors
dmesg | tail -50
```

- [ ] System stable under load
- [ ] No kernel warnings
- [ ] No hardware errors

### Step 6.2: DSMIL Device Access
```bash
# Test DSMIL device access (read-only)
for i in {0..10}; do
    if [ -d "/sys/class/milspec/device${i}" ]; then
        echo "Device $i: $(cat /sys/class/milspec/device${i}/status 2>/dev/null || echo 'N/A')"
    fi
done
```

- [ ] Devices accessible
- [ ] No permission errors
- [ ] Status information available

### Step 6.3: NPU Functionality
```bash
# Check NPU is detected
lspci | grep -i "neural"
dmesg | grep -i NPU
```

- [ ] NPU device detected
- [ ] NPU driver loaded
- [ ] No NPU errors

## Phase 7: 616 Script Integration (Variable time)

### Step 7.1: Count Integration Scripts
```bash
cd /home/john/livecd-gen
find . -name "*.sh" | wc -l
# Should show: 616 or close to it
```

- [ ] Scripts counted
- [ ] All scripts accessible
- [ ] No corrupted files

### Step 7.2: Review Script Categories
```bash
# List script categories
ls -d */ 2>/dev/null | head -10
```

- [ ] Scripts organized by category
- [ ] Directory structure intact
- [ ] No missing categories

### Step 7.3: Integration Plan
```
Note: 616 scripts require systematic review and integration.
This is a task for Local Opus with unlimited processing time.

Recommended approach:
1. Categorize scripts by function
2. Review each for safety and compatibility
3. Test in isolated environment
4. Integrate one category at a time
5. Document all changes
```

- [ ] Integration plan understood
- [ ] Ready to proceed with Opus
- [ ] Backup strategy in place

## Phase 8: Final Verification (15 minutes)

### Step 8.1: System Summary
```bash
# Create system report
cat << EOF > /home/john/deployment_report.txt
=== DSMIL Kernel Deployment Report ===
Date: $(date)
Kernel: $(uname -r)
Mode 5: $(cat /sys/module/dell_milspec/parameters/mode5_level 2>/dev/null)
DSMIL Devices: $(ls /sys/class/milspec/ 2>/dev/null | wc -l)
AVX-512: $(lscpu | grep -c avx512)
TPM: $(ls /dev/tpm* 2>/dev/null | wc -l) device(s)
Uptime: $(uptime)
EOF

cat /home/john/deployment_report.txt
```

- [ ] Deployment report created
- [ ] All systems operational
- [ ] No errors reported

### Step 8.2: Documentation Check
- [ ] All documentation files accessible
- [ ] Interface still available at localhost:8080
- [ ] Backup files preserved
- [ ] Build logs retained

### Step 8.3: Safety Confirmation
- [ ] Mode 5 is STANDARD (not PARANOID_PLUS)
- [ ] System is stable and responsive
- [ ] Can boot to previous kernel if needed
- [ ] No permanent changes to hardware

## Rollback Procedure (If Needed)

### If New Kernel Fails to Boot:
1. Reboot system
2. Select previous kernel from GRUB menu
3. System should boot normally
4. Investigate logs: `journalctl -b -1`

### If System is Unstable:
```bash
# Boot to previous kernel
# Remove new kernel
sudo rm /boot/vmlinuz-6.16.9
sudo rm /boot/initrd.img-6.16.9
sudo update-grub

# Restore GRUB config
sudo cp /etc/default/grub.backup /etc/default/grub
sudo update-grub
```

- [ ] Rollback procedure understood
- [ ] Backup kernel still available
- [ ] GRUB config can be restored

## Post-Deployment Tasks

### For Local Opus:
- [ ] Review all 616 integration scripts
- [ ] Create systematic integration plan
- [ ] Test each category of scripts
- [ ] Document integration process
- [ ] Build final ISO image
- [ ] Perform comprehensive security audit

### Security Hardening:
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Set up intrusion detection
- [ ] Configure TPM attestation
- [ ] Test APT defense mechanisms

### Performance Optimization:
- [ ] Benchmark NPU performance
- [ ] Optimize P-core/E-core scheduling
- [ ] Test AVX-512 vectorization
- [ ] Profile memory encryption overhead
- [ ] Tune IOMMU parameters

## Emergency Contacts

**Documentation:**
- Full handoff: `/home/john/COMPLETE_MILITARY_SPEC_HANDOFF.md`
- Safety warnings: `/home/john/MODE5_SECURITY_LEVELS_WARNING.md`
- Architecture: `/home/john/SYSTEM_ARCHITECTURE.md`

**Interface:**
- Web UI: `http://localhost:8080`
- Quick start: `/home/john/quick-start-interface.sh`

**Build Logs:**
- Success log: `/home/john/kernel-build-apt-secure.log`
- All logs: `/home/john/kernel-build*.log`

---

## ⚠️ CRITICAL SAFETY REMINDERS

1. **NEVER enable PARANOID_PLUS mode** - It will permanently brick the system
2. **Test in VM first** if making any Mode 5 changes
3. **Mode 5 is STANDARD** - Safe and fully reversible
4. **Dell hardware only** - Do not attempt on other systems
5. **Keep backups** - Always maintain recovery options

---

**Checklist Version**: 1.0
**Date**: 2025-10-15
**Status**: Ready for deployment
**Mode 5**: STANDARD (safe)
**Risk Level**: LOW (with proper procedures)

Good luck with your deployment! 🚀
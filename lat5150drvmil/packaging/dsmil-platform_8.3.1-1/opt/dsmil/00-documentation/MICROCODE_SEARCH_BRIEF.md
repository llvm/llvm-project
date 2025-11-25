# MICROCODE 0x1c ACQUISITION BRIEF
**Target:** Intel Meteor Lake-H Core Ultra 7 165H microcode version 0x1c or earlier
**Current Problem:** Microcode 0x24 disables AVX-512 via hardware fuse
**Mission:** Locate and download microcode binary file

---

## SYSTEM SPECIFICATIONS
```
CPU: Intel Core Ultra 7 165H (Meteor Lake-H)
CPUID: Family 6, Model 167, Stepping 1
Microcode Signature: 06-a7-01
Current Version: 0x24 (blocks AVX-512)
Target Version: 0x1c, 0x1e, or any < 0x22
Architecture: x86_64
```

## WHAT WE NEED
**File:** Raw microcode binary for CPUID 06-a7-01
**Size:** Approximately 100-120 KB (based on current 0x24 = 106KB)
**Version:** 0x1c or earlier (before Intel disabled AVX-512)
**Format:** Raw binary blob (no header, no compression)

## WHERE TO SEARCH

### Primary Sources
1. **Intel Official Archive**
   - https://github.com/intel/Intel-Linux-Processor-Microcode-Data-Files
   - Look in releases from 2023-Q4 or earlier
   - File path in repo: intel-ucode/06-a7-01
   - Check commit history for version 0x1c

2. **Linux Firmware Git History**
   - https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git
   - Path: intel-ucode/06-a7-01
   - Browse history: 2023-2024 commits
   - Download from commit before 0x24 update

3. **Debian Package Archive**
   - Package: intel-microcode
   - Version: Look for 3.20230808.x or earlier (pre-2024)
   - Extract: dpkg-deb -x intel-microcode_*.deb /tmp/extract
   - File: lib/firmware/intel-ucode/06-a7-01

4. **Archive.org Snapshots**
   - Search for archived versions of Intel microcode repos
   - Look for snapshots from 2023

### Alternative Sources
5. **Linux Distribution Archives**
   - Arch Linux Archive: https://archive.archlinux.org/packages/i/intel-ucode/
   - Fedora koji: https://koji.fedoraproject.org/koji/packageinfo?packageID=5007
   - Ubuntu Launchpad: Old intel-microcode versions

6. **Vendor BIOS Images**
   - Dell Latitude 5450 BIOS version 1.0.x - 1.10.x
   - Extract BIOS using UEFITool
   - Search for microcode module inside

## VERIFICATION
Once file obtained, verify:
```bash
# Check size (should be ~100-120KB)
ls -lh 06-a7-01

# Check it's not empty
file 06-a7-01

# Should show: data (binary blob)
```

## INSTALLATION INSTRUCTIONS
```bash
# Backup current (already done)
sudo cp /lib/firmware/intel-ucode/06-a7-01.BACKUP /root/microcode-0x24.backup

# Install old microcode
sudo cp 06-a7-01 /lib/firmware/intel-ucode/06-a7-01

# Update initramfs
sudo update-initramfs -u

# Reboot
sudo reboot

# Verify after boot
grep microcode /proc/cpuinfo | head -1
# Should show: microcode: 0x1c (or 0x1e, 0x20, anything < 0x22)
```

## SUCCESS CRITERIA
After reboot with microcode 0x1c:
```bash
# Load DSMIL module
sudo insmod /home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko

# Unlock
echo unlock | sudo tee /proc/dsmil_avx512

# Verify AVX-512 visible
cat /proc/cpuinfo | grep avx512
# Should show: avx512f avx512dq avx512cd avx512bw avx512vl avx512_vbmi ...
```

## CRITICAL NOTES
- Meteor Lake-H (06-a7-01) is NEW architecture (2023-2024)
- Microcode history is SHORT - may only have 3-5 versions total
- Version 0x1c might not exist publicly if it was OEM-only
- Intel may have never released AVX-512-enabled version publicly
- BIOS version 1.17.2 ships with 0x24 (confirmed)
- Earlier BIOS versions (1.0.x - 1.15.x) might have older microcode

## FALLBACK OPTIONS IF NOT FOUND
1. Dell BIOS downgrade to 1.0.x (risky, may lose features)
2. Contact Dell support for OEM microcode access
3. Check vx-underground or firmware archives
4. Accept AVX2/AVX_VNNI only (66.4 TOPS AI still functional)

## DELIVERY
Upload found file to:
- Direct download link, OR
- Git repo commit hash + instructions, OR
- Package name + version for apt/dnf/pacman

---

**Priority:** HIGH
**Urgency:** Medium (system functional without, but 60% performance gain if successful)
**Risk:** Low (can restore 0x24 backup if issues occur)

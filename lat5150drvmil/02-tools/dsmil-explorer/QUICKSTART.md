# DSMIL Automation Suite - Quick Start Guide

**Get started in 5 minutes!**

## Step 1: Load Kernel Module

```bash
cd /home/user/LAT5150DRVMIL/01-source/kernel
sudo insmod dsmil-72dev.ko

# Verify
lsmod | grep dsmil
ls -l /dev/dsmil0
```

## Step 2: Quick Device Scan (No Hardware Access - Safe)

```bash
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-explorer

# Scan all devices
python3 dsmil_scanner.py --quick
```

**Output:**
```
Total devices: 84
Accessible: 79
By Risk Level:
  MONITORED        6 devices (  7.1%)
  QUARANTINED      5 devices (  6.0%)
  UNKNOWN         67 devices ( 79.8%)
  RISKY            6 devices (  7.1%)
```

## Step 3: Probe Your First Device (Safe)

```bash
# Probe device 0x8030 (Group 3 - Data Processing)
# Phase 1 only - just reads capabilities
sudo python3 dsmil_probe.py --device 0x8030 --phase 1
```

**Output:**
```
[14:23:45] INFO     probe        [0x8030] Phase 1: RECONNAISSANCE
[14:23:45] INFO     probe        [0x8030] Capabilities read successfully
[14:23:45] INFO     operation    [0x8030] Operation 'phase1_reconnaissance' succeeded

Probe Results for Device 0x8030
================================================================================
Success: True
Phases completed: RECONNAISSANCE
Errors: 0
Warnings: 0
```

## Step 4: Probe a Full Group (12 Devices)

```bash
# Probe entire Group 3 with Phase 2 (passive observation)
sudo python3 dsmil_probe.py --range 0x8030 0x803B --phase 2
```

**This will:**
- Probe all 12 devices in Group 3 (Data Processing)
- Read capabilities and status
- Monitor for 3 seconds each
- Detect any anomalies
- Generate comprehensive logs

**Time**: ~40 seconds

## Step 5: Generate Documentation

```bash
# Export scan results
python3 dsmil_scanner.py --quick --export output/scan_results.json

# Generate docs
python3 dsmil_docgen.py --input output/scan_results.json

# View
ls output/docs/
cat output/docs/DEVICE_INDEX.md
```

## That's It!

You now have:
✅ Scanned all 84 devices
✅ Probed 12 unknown devices safely
✅ Generated comprehensive documentation
✅ Collected baseline data for future exploration

## Next Steps

### Option A: Continue with Group 2 (Network)
```bash
sudo python3 dsmil_probe.py --range 0x8020 0x802B --phase 2
```

### Option B: Start Real-Time Monitoring
```bash
# In separate terminal
sudo python3 dsmil_monitor.py
```

### Option C: Deep Dive on Specific Device
```bash
# Read the generated profile
cat output/docs/device_0x8030_profile.md
```

## Safety Notes

✅ **Always Safe:**
- `python3 dsmil_scanner.py --quick` (no hardware access)
- `python3 dsmil_probe.py --device 0xXXXX --phase 1` (read-only)
- `python3 dsmil_probe.py --device 0xXXXX --dry-run` (simulation)

⚠️ **Caution Required:**
- `--phase 2` accesses hardware (but still read-only and safe)
- Unknown devices should be probed incrementally

🔴 **Never Access:**
- 0x8009, 0x800A, 0x800B (data destruction)
- 0x8019, 0x8029 (network kill)
- These are automatically blocked by the safety system

## Common Commands

```bash
# Scan all devices (quick)
python3 dsmil_scanner.py --quick

# Scan with hardware access
sudo python3 dsmil_scanner.py

# Probe single device (Phase 1)
sudo python3 dsmil_probe.py --device 0x8030

# Probe range (Phase 2)
sudo python3 dsmil_probe.py --range 0x8030 0x803B --phase 2

# Monitor system
sudo python3 dsmil_monitor.py

# Generate docs
python3 dsmil_docgen.py --input results.json
```

## Troubleshooting

**Problem:** Module not loaded
```bash
sudo insmod /home/user/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko
```

**Problem:** Permission denied
```bash
# Run with sudo
sudo python3 dsmil_probe.py ...
```

**Problem:** Device not responding
```bash
# Check kernel messages
dmesg | tail -20
```

## Get Help

```bash
python3 dsmil_probe.py --help
python3 dsmil_scanner.py --help
python3 dsmil_monitor.py --help
python3 dsmil_docgen.py --help
```

---

**Ready to explore the remaining 67 devices? Let's go! 🚀**

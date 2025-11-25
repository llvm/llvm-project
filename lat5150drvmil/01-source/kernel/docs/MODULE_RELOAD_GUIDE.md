# DSMIL Module Reload Guide

## Overview

This guide explains how to handle the "Driver already registered" error that can occur when rebuilding and reloading the DSMIL kernel module.

## The Problem

When you see this error in dmesg:
```
Error: Driver 'dsmil-72dev' is already registered, aborting...
dsmil-84dev: Failed to register platform driver
```

And modprobe/insmod returns:
```
modprobe: ERROR: could not insert 'dsmil_84dev': Device or resource busy
```

This means an old DSMIL module is still registered in the kernel, preventing the new module from loading.

## Why This Happens

1. **Platform Driver Persistence**: Even after `rmmod`, the platform driver may remain registered in some kernel states
2. **Module Dependencies**: Other kernel components may hold references to the old module
3. **Incomplete Unload**: The module's exit function may not have fully completed
4. **Device Node Locks**: Applications may still have `/dev/dsmil*` nodes open

## Solutions

### Solution 1: Use the Enhanced Build Script (Automatic)

The updated `build-and-install.sh` now automatically:
- Checks both `lsmod` and `/sys/module` for existing modules
- Unloads all DSMIL module variants (dsmil_84dev, dsmil_72dev)
- Scans for any other DSMIL-related modules
- Waits for the kernel to fully cleanup after unload
- Provides helpful error messages if the module can't be unloaded

Simply run:
```bash
cd /path/to/LAT5150DRVMIL/01-source/kernel
sudo ./build-and-install.sh
```

### Solution 2: Use the Standalone Reload Script

For quick module reloads without rebuilding:
```bash
cd /path/to/LAT5150DRVMIL/01-source/kernel
sudo ./reload-dsmil-module.sh
```

This script:
- Unloads all DSMIL module variants
- Checks `/sys/module` for lingering entries
- Reloads the module
- Verifies the module is loaded correctly
- Shows device node status

### Solution 3: Manual Module Management

If the automated scripts don't work, try these steps manually:

#### Step 1: Check what's loaded
```bash
lsmod | grep -i dsmil
ls -la /sys/module | grep dsmil
```

#### Step 2: Unload all DSMIL modules
```bash
sudo rmmod dsmil_72dev  # Legacy module name
sudo rmmod dsmil_84dev  # New module name
```

#### Step 3: Verify unload
```bash
lsmod | grep -i dsmil   # Should return nothing
```

#### Step 4: Check for stuck modules
```bash
ls -la /sys/module | grep dsmil  # Should return nothing
```

#### Step 5: Force unload (if needed)
```bash
sudo rmmod -f dsmil_72dev
sudo rmmod -f dsmil_84dev
```

#### Step 6: Close any open device handles
```bash
# Find processes using DSMIL devices
sudo lsof | grep dsmil

# Kill the processes if needed
sudo kill -9 <PID>
```

#### Step 7: Reload the module
```bash
cd /path/to/LAT5150DRVMIL/01-source/kernel
sudo modprobe dsmil_84dev
# OR
sudo insmod dsmil-84dev.ko
```

### Solution 4: System Reboot (Last Resort)

If all else fails, a system reboot will clear all kernel module state:
```bash
sudo reboot
```

After reboot, rebuild and install:
```bash
cd /path/to/LAT5150DRVMIL/01-source/kernel
sudo ./build-and-install.sh
```

## Prevention

To avoid this issue in the future:

1. **Always unload before rebuild**: The build script now does this automatically
2. **Close applications first**: Stop the Control Center and any other DSMIL-using applications before rebuilding
3. **Use the reload script**: For quick testing, use `reload-dsmil-module.sh` instead of rebuilding

## Diagnostic Commands

### Check module status
```bash
# Show loaded modules
lsmod | grep -i dsmil

# Show module info
modinfo dsmil_84dev

# Show module parameters
ls -la /sys/module/dsmil_84dev/parameters/
```

### Check device nodes
```bash
# List DSMIL device nodes
ls -la /dev/dsmil*

# Check permissions
stat /dev/dsmil-72dev
```

### Check kernel messages
```bash
# Recent DSMIL messages
dmesg | grep -i dsmil | tail -20

# Follow kernel log in real-time
dmesg -w | grep -i dsmil
```

### Check for stuck processes
```bash
# Find processes using DSMIL
sudo lsof | grep dsmil
sudo fuser /dev/dsmil-72dev
```

## Error Messages Explained

### "Driver already registered"
The platform driver name is already in use. Unload the old module first.

### "Device or resource busy"
Either the device is in use by an application, or the old module wasn't fully unloaded.

### "No such device"
The module loaded but didn't create device nodes. Check dmesg for initialization errors.

### "Operation not permitted"
You need root privileges. Run with `sudo`.

## Integration with Control Center

The `launch-dsmil-control-center.sh` script automatically runs the enhanced `build-and-install.sh`, which now handles module reloading properly.

If the launcher reports:
```
⚠ Driver build/install failed
  Control Center will work in simulation mode
```

Check the kernel messages:
```bash
dmesg | tail -20
```

And manually unload/reload:
```bash
sudo ./reload-dsmil-module.sh
```

Then re-launch the Control Center.

## Quick Reference

| Task | Command |
|------|---------|
| Build & Install | `sudo ./build-and-install.sh` |
| Reload Module | `sudo ./reload-dsmil-module.sh` |
| Unload Module | `sudo rmmod dsmil_84dev` |
| Load Module | `sudo modprobe dsmil_84dev` |
| Check Status | `lsmod \| grep dsmil` |
| View Logs | `dmesg \| grep -i dsmil` |
| Force Unload | `sudo rmmod -f dsmil_84dev` |

## Additional Resources

- **Build Script**: `01-source/kernel/build-and-install.sh`
- **Reload Script**: `01-source/kernel/reload-dsmil-module.sh`
- **Launcher**: `launch-dsmil-control-center.sh`
- **Kernel Logs**: `dmesg -w`
- **Module Info**: `modinfo dsmil_84dev`

## Troubleshooting Flowchart

```
Module won't load?
    ↓
Check: Is old module loaded? (lsmod | grep dsmil)
    ↓ YES → sudo rmmod dsmil_72dev && sudo rmmod dsmil_84dev
    ↓ NO → Continue
    ↓
Check: Module in /sys/module? (ls /sys/module | grep dsmil)
    ↓ YES → sudo rmmod -f dsmil_84dev
    ↓ NO → Continue
    ↓
Check: Processes using device? (lsof | grep dsmil)
    ↓ YES → Kill processes or close applications
    ↓ NO → Continue
    ↓
Try: sudo ./reload-dsmil-module.sh
    ↓ FAILS → Check dmesg for specific errors
    ↓ SUCCESS → Module loaded!
```

---

**Last Updated**: 2025-01-11
**Version**: 1.0.0
**Author**: LAT5150DRVMIL AI Platform

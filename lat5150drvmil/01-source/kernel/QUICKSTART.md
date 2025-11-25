# DSMIL Educational Driver - Quick Start

## 🚀 Get Started in 2 Minutes!

This is a **working, simplified kernel driver** for learning. No hardware needed!

## Prerequisites

```bash
# Make sure you have kernel headers
sudo apt-get install linux-headers-$(uname -r)

# Or on other distros
sudo yum install kernel-devel
```

## Build and Test

### Option 1: Automated (Recommended)

```bash
cd 01-source/kernel

# One command does everything:
sudo make -f Makefile.simple reload
```

This will:
1. Build the module
2. Load it
3. Create /dev/dsmil
4. Run the test program

### Option 2: Step by Step

```bash
cd 01-source/kernel

# 1. Build
make -f Makefile.simple

# 2. Load
sudo make -f Makefile.simple load

# 3. Test
cat /dev/dsmil
echo "reset" > /dev/dsmil
cat /sys/class/misc/dsmil/version

# 4. Run test program
make -f Makefile.simple test
```

## What You'll See

```
╔══════════════════════════════════════════════════╗
║  DSMIL Educational Driver Demonstration          ║
╚══════════════════════════════════════════════════╝

=== Testing read() ===
Read 103 bytes:
DSMIL Educational Driver v1.0-educational
Devices: 84
Access count: 1
Status: Ready

=== Testing write() ===
Writing: Hello DSMIL!
Wrote 13 bytes

=== Testing ioctl read (device 0x8000) ===
Device 0x8000 simulated value: 0x80001234

[... more tests ...]

╔══════════════════════════════════════════════════╗
║   Demonstration Complete!                        ║
╚══════════════════════════════════════════════════╝
```

## Play Around

```bash
# Read device status
cat /dev/dsmil

# Send commands
echo "reset" > /dev/dsmil
echo "hello" > /dev/dsmil

# Check sysfs attributes
ls /sys/class/misc/dsmil/
cat /sys/class/misc/dsmil/device_count
cat /sys/class/misc/dsmil/access_count

# Reset counter via sysfs
echo 0 > /sys/class/misc/dsmil/access_count

# Watch kernel logs
dmesg | grep dsmil-sim

# Watch logs in real-time
sudo dmesg -w | grep dsmil-sim
```

## Test Program Usage

```bash
# Full demo
sudo ./examples/test-dsmil demo

# Individual commands
sudo ./examples/test-dsmil read
sudo ./examples/test-dsmil write "Test message"
sudo ./examples/test-dsmil ioctl-read 0x8000
sudo ./examples/test-dsmil count
sudo ./examples/test-dsmil sysfs
```

## Clean Up

```bash
# Unload module
sudo make -f Makefile.simple unload

# Clean build files
make -f Makefile.simple clean
```

## What Makes This Different?

### ✅ **This Works** (Simplified Version)

- Creates `/dev/dsmil` automatically
- All operations return real data
- Sysfs attributes work
- Test program included
- Works without hardware

### ⚠️ **Original Doesn't** (Complex Version)

- Needs platform device binding
- Many functions return -ENOTSUPP
- More complex (5500+ lines)
- Harder to learn from

## File Structure

```
01-source/kernel/
├── core/
│   ├── dsmil-simple.c          ← Simplified driver (450 lines)
│   └── dsmil-72dev.c           ← Original complex version
├── examples/
│   └── test-dsmil.c            ← Test program
├── Makefile.simple             ← Build system (easy)
├── Makefile                    ← Original makefile (complex)
├── QUICKSTART.md               ← This file
└── EDUCATIONAL_IMPROVEMENTS.md ← Detailed comparison
```

## Learning Path

### Day 1: Run It
```bash
# Just make it work
sudo make -f Makefile.simple reload
```

### Day 2: Explore It
```bash
# Play with /dev/dsmil and sysfs
cat /dev/dsmil
echo test > /dev/dsmil
ls /sys/class/misc/dsmil/
```

### Day 3: Read the Code
```bash
# Study the simplified driver
less core/dsmil-simple.c
```

### Day 4: Modify It
```bash
# Add your own ioctl command
# Modify simulation layer
# Add sysfs attribute
```

### Day 5: Compare
```bash
# See the difference in complexity
wc -l core/dsmil-simple.c   # 450 lines
wc -l core/dsmil-72dev.c    # 5500+ lines
```

## Common Issues

### "Failed to open /dev/dsmil"
```bash
# Module not loaded
sudo make -f Makefile.simple load
```

### "Module not found"
```bash
# Not built yet
make -f Makefile.simple
```

### Permission denied on /dev/dsmil
```bash
# Module sets permissions to 0666 (rw-rw-rw-)
# But you might need sudo for write operations
sudo echo test > /dev/dsmil
```

### Can't compile - missing headers
```bash
# Install kernel headers
sudo apt-get install linux-headers-$(uname -r)
```

## Next Steps

1. **Run the quick start** above
2. **Read** [EDUCATIONAL_IMPROVEMENTS.md](EDUCATIONAL_IMPROVEMENTS.md) for details
3. **Study** `core/dsmil-simple.c` source code
4. **Experiment** - modify the code!
5. **Compare** with complex version `core/dsmil-72dev.c`

## Help

```bash
# See all available commands
make -f Makefile.simple help

# Get module info
sudo modinfo dsmil-simple.ko

# Check if loaded
lsmod | grep dsmil_simple

# View all kernel messages
dmesg | grep dsmil-sim
```

## One-Liners

```bash
# Build, load, test (full cycle)
sudo make -f Makefile.simple reload

# Quick status check
cat /dev/dsmil

# View all sysfs attributes
cat /sys/class/misc/dsmil/*

# Reset and check
echo 0 > /sys/class/misc/dsmil/access_count && cat /sys/class/misc/dsmil/access_count
```

---

**Happy Learning! 🎓**

This driver is designed to be your first kernel module. It's simple, it works, and it teaches real concepts.

For detailed information, see [EDUCATIONAL_IMPROVEMENTS.md](EDUCATIONAL_IMPROVEMENTS.md)

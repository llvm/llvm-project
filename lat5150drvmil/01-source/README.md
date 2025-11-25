# 💻 Source Code Directory - Navigation Guide

## 🧭 **WHERE AM I?**
You are in: `/01-source/` - All source code and implementation

## 🏠 **NAVIGATION**
```bash
# Back to project root
cd ..
# or
cd /opt/scripts/milspec/
```

## 🗺️ **MASTER REFERENCES**
- Root Navigation: `../MASTER-NAVIGATION.md`
- Execution Flow: `../EXECUTION-FLOW.md`
- Implementation Plans: `../00-documentation/01-planning/`

## 📁 **SOURCE CODE ORGANIZATION**

### **kernel-driver/** - Main Kernel Module
```yaml
dell-millspec-enhanced.c    # Main driver (85KB, 1600+ lines)
dell-milspec.h             # Public API header
dell-milspec-internal.h    # Internal definitions
dell-milspec-regs.h        # Hardware register map
dell-milspec-crypto.h      # Crypto operations
dell-smbios-local.h        # SMBIOS interface
Makefile                   # Build configuration
Kconfig                    # Kernel config options
dell-milspec.ko            # Compiled module (85KB)
```

### **userspace-tools/** - Control Utilities
```yaml
milspec-control.c          # CLI control utility (12KB)
milspec-monitor.c          # Event monitoring daemon (13KB)
milspec-events.c           # Simple event watcher
milspec-control.1          # Man page
milspec-monitor.1          # Man page
milspec-completion.bash    # Shell completion
```

### **tests/** - Test Suite
```yaml
test-milspec.c             # IOCTL test program
test-milspec               # Compiled test binary
test-utils.sh              # Test helper scripts
```

### **scripts/** - Utility Scripts
```yaml
enumeration.sh             # Hardware enumeration
examples.sh                # Usage examples
install-utils.sh           # Installation helper
milspec-analysis.sh        # System analysis
```

### **systemd/** - Service Files
```yaml
dell-milspec.service       # Systemd service unit
```

## 🚀 **QUICK BUILD COMMANDS**

### **Build Kernel Module**
```bash
cd kernel-driver/
make clean && make
# Output: dell-milspec.ko
```

### **Build Userspace Tools**
```bash
cd userspace-tools/
gcc -o milspec-control milspec-control.c
gcc -o milspec-monitor milspec-monitor.c
```

### **Run Tests**
```bash
cd tests/
./test-utils.sh
sudo ./test-milspec
```

## 📊 **CODE STATISTICS**

```yaml
Kernel Module:
  - Main Driver: 1600+ lines
  - Module Size: 85KB
  - Features: Mode 5, DSMIL, GPIO, TPM

Userspace Tools:
  - Control CLI: 12KB
  - Monitor Daemon: 13KB
  - Total Tools: 6 utilities

Tests:
  - IOCTL Tests: Complete
  - Integration: Pending
```

## 🎯 **DEVELOPER QUICK REFERENCE**

### **Key APIs**
```c
// Main IOCTL commands (dell-milspec.h)
MILSPEC_IOC_GET_STATUS
MILSPEC_IOC_SET_MODE5
MILSPEC_IOC_ACTIVATE_DSMIL
MILSPEC_IOC_EMERGENCY_WIPE

// Hardware registers (dell-milspec-regs.h)
MILSPEC_REG_STATUS
MILSPEC_REG_CONTROL
MILSPEC_REG_MODE5_LEVEL
```

### **Build Flags**
```makefile
# AVX-512 optimizations
EXTRA_CFLAGS += -mavx512f -O3

# Debug build
make DEBUG=1
```

## 🔗 **RELATED DOCUMENTATION**

- **Implementation Plans**: `../00-documentation/01-planning/`
- **API Reference**: `../00-documentation/05-reference/api/`
- **Build Notes**: `../00-documentation/05-reference/build-notes.md`
- **Testing Guide**: `../00-documentation/01-planning/phase-1-core/TESTING-INFRASTRUCTURE-PLAN.md`

## ⚡ **COMMON TASKS**

### **Install Module**
```bash
cd kernel-driver/
sudo insmod dell-milspec.ko
sudo lsmod | grep milspec
```

### **Test Module**
```bash
cd tests/
sudo ./test-milspec
dmesg | tail -50
```

### **Monitor Events**
```bash
cd userspace-tools/
sudo ./milspec-monitor
```

## 📝 **DEVELOPMENT WORKFLOW**

1. **Read Plan**: Check relevant plan in `../00-documentation/01-planning/`
2. **Edit Code**: Make changes in appropriate directory
3. **Build**: Use make or gcc
4. **Test**: Run tests from `tests/`
5. **Document**: Update relevant docs

---
**Remember**: Always check the implementation plan before coding!
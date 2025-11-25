# AVX-512 Unlock for Intel Meteor Lake (Core Ultra 7 165H)

**Complete toolkit to unlock AVX-512 on Dell Latitude 5450 Covert Edition**

---

## 🚀 Two Methods Available

### 🆕 **Method 1: Advanced (Recommended) - Keep E-cores Enabled**
```bash
# Enable AVX-512 while keeping all cores active
sudo ./unlock_avx512_advanced.sh enable

# If that doesn't work, use microcode fallback
sudo ./unlock_avx512_advanced.sh microcode-fallback
sudo reboot

# Verify it worked
./verify_avx512_advanced.sh

# Run your AVX-512 programs pinned to P-cores
run-on-pcores ./myapp
# or
taskset -c 0-5 ./myapp
```

**Benefits:**
- ✅ Keep all 16 cores (6 P-cores + 10 E-cores)
- ✅ Better multitasking performance
- ✅ AVX-512 available on P-cores via task pinning
- ✅ DSMIL driver integration for Dell platform features

### 📌 **Method 2: Traditional - Disable E-cores**
```bash
# Disable E-cores to unlock AVX-512
sudo ./unlock_avx512.sh enable

# Verify it worked
./verify_avx512.sh

# Source AVX-512 compiler flags
source ./avx512_compiler_flags.sh

# Compile with AVX-512
gcc $CFLAGS_AVX512 -o myapp myapp.c $LDFLAGS_AVX512
```

**Benefits:**
- ✅ Simpler approach (no task pinning needed)
- ✅ Works on older kernels
- ❌ Lose 10 E-cores (drops from 16 to 6 cores)

---

## ⚡ Quick Start (Advanced Method)

```bash
# 1. Enable AVX-512 with advanced method
sudo ./unlock_avx512_advanced.sh enable

# 2. Verify it worked
./verify_avx512_advanced.sh

# 3. Source AVX-512 compiler flags
source ./avx512_compiler_flags.sh

# 4. Compile with AVX-512
gcc $CFLAGS_AVX512 -o myapp myapp.c $LDFLAGS_AVX512

# 5. Run on P-cores only (AVX-512 capable)
run-on-pcores ./myapp
```

### DSMIL Driver Helpers

The DSMIL kernel driver (canonical device `/dev/dsmil-84dev`, legacy alias
`/dev/dsmil-72dev`) now exposes sysfs knobs and an ioctl helper so you can
verify service mode or rerun the unlock without touching procfs:

```
cat /sys/class/dsmil-84dev/dsmil-84dev/service_mode
echo 1 | sudo tee /sys/class/dsmil-84dev/dsmil-84dev/service_mode_override
echo 1 | sudo tee /sys/class/dsmil-84dev/dsmil-84dev/avx512_unlock
# Legacy alias (still exported for existing tooling)
# /sys/class/dsmil-72dev/dsmil-72dev/{service_mode,service_mode_override,avx512_unlock}
sudo ./scripts/dsmil_ioctl_tool.py msr --msr 0x1a0 --cpu 0
sudo ./scripts/dsmil_ioctl_tool.py smi --command 0xA512 --cpu 0xff
````

SMI failures now return readable status codes, so experimentation is safer.

---

## 📋 Background

### The Problem

Intel Meteor Lake (Core Ultra 7 165H) has a **hybrid architecture**:
- **6 P-cores**: Support AVX-512 (full SIMD capability)
- **10 E-cores**: No AVX-512 support (only up to AVX2)

Intel **disabled AVX-512** by microcode on Meteor Lake because:
1. Mixing AVX-512 (P-cores) and AVX2 (E-cores) causes frequency scaling issues
2. Thread migration between P and E-cores would crash on AVX-512 instructions
3. Simpler to disable than fix scheduler

### The Solution

**Disable E-cores** → P-cores can use AVX-512

**Trade-off**:
- ✅ **GAIN**: AVX-512 vectorization (2x wider than AVX2)
- ✅ **GAIN**: 15-40% faster for vectorizable workloads
- ❌ **LOSS**: 10 E-cores for background tasks
- ❌ **LOSS**: Total core count drops from 16 to 6

**Best for**: Kernel compilation, scientific computing, cryptography, AI inference

---

## 📂 Files Included

| File | Purpose |
|------|---------|
| `unlock_avx512_advanced.sh` | 🆕 Advanced: Enable AVX-512 while keeping E-cores active |
| `verify_avx512_advanced.sh` | 🆕 Advanced verification with P-core task pinning tests |
| `unlock_avx512.sh` | Traditional: Enable/disable AVX-512 by toggling E-cores |
| `verify_avx512.sh` | Traditional: 5-test verification suite for AVX-512 |
| `avx512_compiler_flags.sh` | Complete AVX-512 optimized compiler flags |
| `README.md` | This file |

---

### Building the DSMIL AVX-512 Enabler Module

Some flows (microcode 0x24 in particular) require the kernel helper module
`dsmil_avx512_enabler.ko`. We now keep its source in-tree, so you can build it
directly from this repository:

```bash
# Build the module against the running kernel
./scripts/build_dsmil_avx512_enabler.sh

# Optionally install to /lib/modules/.../extra
./scripts/build_dsmil_avx512_enabler.sh --install

# Load the module and trigger the unlock
sudo insmod 01-source/drivers/dsmil_avx512_enabler/dsmil_avx512_enabler.ko
echo unlock | sudo tee /proc/dsmil_avx512
```

This replaces the older instructions that referenced the external
`/home/john/livecd-gen/kernel-modules` path.

---

## 🔧 Advanced Method Explained

### How It Works

The advanced method uses three approaches in order of preference:

1. **MSR-based Enable (Experimental)**
   - Attempts to enable AVX-512 via Model Specific Registers on P-cores
   - May not work if microcode enforces disable
   - No reboot required

2. **CPU Affinity Task Pinning (Recommended)**
   - Pins AVX-512 workloads to P-cores only (CPUs 0-5)
   - Uses `taskset` or custom `run-on-pcores` wrapper
   - E-cores stay active for non-AVX-512 tasks
   - Requires discipline: all AVX-512 code must be pinned

3. **Microcode Disable Fallback (Nuclear Option)**
   - Disables OS microcode updates via GRUB parameters
   - Forces CPU to use BIOS/UEFI microcode
   - BIOS microcode may not have AVX-512 disabled
   - Requires reboot

### DSMIL Driver Integration

The advanced method automatically loads the DSMIL (Dell System Military Integration Layer) driver for enhanced Dell platform features:

- Dell WMI/SMBIOS integration
- Hardware token validation
- NPU acceleration support
- Military-spec thermal profiles
- Enhanced security features

This provides better integration with Dell Latitude 5450 MIL-SPEC hardware.

---

## 🚀 Usage Guide

### 1. Unlock AVX-512 (Advanced Method)

```bash
# Step 1: Try advanced enable (keeps E-cores active)
sudo ./unlock_avx512_advanced.sh enable

# Step 2: Check if it worked
./verify_avx512_advanced.sh

# Step 3: If AVX-512 not detected, use microcode fallback
sudo ./unlock_avx512_advanced.sh microcode-fallback
sudo reboot  # Required after microcode changes

# Step 4: Verify after reboot
./verify_avx512_advanced.sh

# Check status anytime
sudo ./unlock_avx512_advanced.sh status
```

**Restore microcode loading**:
```bash
sudo ./unlock_avx512_advanced.sh restore-microcode
sudo reboot
```

### 1b. Unlock AVX-512 (Traditional Method)

```bash
# Enable AVX-512 (disable E-cores)
sudo ./unlock_avx512.sh enable

# Check status
sudo ./unlock_avx512.sh status

# Disable AVX-512 (re-enable E-cores)
sudo ./unlock_avx512.sh disable
```

**Make it persistent across reboots**:
```bash
sudo ./unlock_avx512.sh enable
# When prompted, type 'y' to create systemd service

# Or manually:
sudo ./unlock_avx512.sh persistent
```

**Remove persistence**:
```bash
sudo ./unlock_avx512.sh remove-persistent
```

---

### 2. Verify AVX-512 is Working

**Advanced method verification**:
```bash
./verify_avx512_advanced.sh
```

**Expected output (Advanced)**:
```
✓ PASS: AVX-512 flags found in /proc/cpuinfo
✓ PASS: All E-cores (6-15) are ONLINE
✓ PASS: AVX-512 program executed successfully on P-core
✓ PASS: run-on-pcores wrapper exists
✓ PASS: AVX-512 instructions (zmm registers) found in binary

[✓] AVX-512 Status: OPERATIONAL
    Advanced method working with P-core pinning
```

**Traditional method verification**:
```bash
./verify_avx512.sh
```

**Expected output (Traditional)**:
```
✓ PASS: AVX-512 flags found in /proc/cpuinfo
✓ PASS: All E-cores (6-15) are disabled
✓ PASS: Compilation successful
✓ PASS: AVX-512 program executed successfully
✓ PASS: AVX-512 instructions (zmm registers) found in binary

[✓] AVX-512 Status: OPERATIONAL
```

---

### 2b. Running AVX-512 Programs (Advanced Method Only)

**CRITICAL**: When using the advanced method, you MUST pin AVX-512 programs to P-cores:

```bash
# Option 1: Use the wrapper (recommended)
run-on-pcores ./my-avx512-program

# Option 2: Use taskset directly
taskset -c 0-5 ./my-avx512-program

# Option 3: Pin to specific P-core
taskset -c 0 ./my-avx512-program  # Run on CPU 0 only
```

**For long-running services**, create a systemd service with CPU affinity:

```ini
[Service]
ExecStart=/usr/bin/my-avx512-service
CPUAffinity=0-5  # Pin to P-cores only
```

**For OpenMP programs**:
```bash
export GOMP_CPU_AFFINITY="0-5"
export OMP_NUM_THREADS="6"
export OMP_PROC_BIND="true"
./my-openmp-avx512-program
```

⚠️ **WARNING**: If you run AVX-512 code on E-cores (CPUs 6-15), it will crash with "Illegal instruction"!

---

### 3. Use AVX-512 Compiler Flags

```bash
# Source the flags
source ./avx512_compiler_flags.sh

# Compile C/C++ programs
gcc $CFLAGS_AVX512 -o app app.c $LDFLAGS_AVX512

# Compile kernel
cd /usr/src/linux
make -j6 KCFLAGS="$KCFLAGS_AVX512" KCPPFLAGS="$KCPPFLAGS_AVX512"

# Use helper functions
compile_avx512 app.c -o app
compile_kernel_avx512
```

---

### 4. Benchmark AVX-512 vs AVX2

```bash
source ./avx512_compiler_flags.sh
benchmark_avx512_vs_avx2
```

**Example output**:
```
AVX2 Result: Time: 0.3420 seconds
AVX-512 Result: Time: 0.2150 seconds

→ AVX-512 is 37% faster
```

---

## 🎯 Compiler Flags Reference

### Complete AVX-512 Flag Set

```bash
export CFLAGS_AVX512="-O3 -pipe -march=meteorlake -mtune=meteorlake \
    -mavx -mavx2 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl \
    -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg \
    -mavx512vpopcntdq -mfma -mavxvnni -maes -msha -flto=auto"
```

### Kernel Flags (Minimal Set)

```bash
export KCFLAGS_AVX512="-O3 -march=meteorlake -mtune=meteorlake \
    -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni \
    -falign-functions=32"
```

### Performance Profiles

```bash
# Maximum Speed (unsafe math)
source ./avx512_compiler_flags.sh
gcc $CFLAGS_AVX512_SPEED -o app app.c

# Balanced (safe)
gcc $CFLAGS_AVX512_BALANCED -o app app.c

# Security Hardened
gcc $CFLAGS_AVX512_SECURE -o app app.c
```

---

## 🧪 Testing AVX-512

### Quick Test

```bash
source ./avx512_compiler_flags.sh
test_avx512
```

### Manual Test

```bash
# Create test program
cat > test.c <<EOF
#include <immintrin.h>
#include <stdio.h>

int main() {
    __m512i a = _mm512_set1_epi32(42);
    __m512i b = _mm512_set1_epi32(8);
    __m512i c = _mm512_add_epi32(a, b);

    int result[16];
    _mm512_storeu_si512((__m512i*)result, c);

    printf("Result: %d\n", result[0]); // Should be 50
    return 0;
}
EOF

# Compile with AVX-512
gcc -O3 -march=meteorlake -mavx512f -mavx512dq test.c -o test

# Run (must use P-core)
taskset -c 0 ./test
```

**Check for AVX-512 instructions**:
```bash
objdump -d test | grep zmm
# Should show instructions like: vmovdqu32 %zmm0, ...
```

---

## ⚙️ Kernel Compilation

### Full Kernel Build with AVX-512

```bash
# 1. Unlock AVX-512
sudo ./unlock_avx512.sh enable

# 2. Source flags
source ./avx512_compiler_flags.sh

# 3. Configure kernel
cd /usr/src/linux-6.x
make menuconfig

# In menuconfig:
# Processor type and features → Processor family → Core2/newer Xeon
# Enable: CONFIG_MARCH_METEORLAKE=y (if available)
# Enable: CONFIG_CC_OPTIMIZE_FOR_PERFORMANCE_O3=y

# 4. Build with AVX-512
make -j6 KCFLAGS="$KCFLAGS_AVX512" KCPPFLAGS="$KCPPFLAGS_AVX512"

# 5. Install
sudo make modules_install
sudo make install
```

### Recommended Kernel Config

```bash
CONFIG_MARCH_METEORLAKE=y
CONFIG_GENERIC_CPU=n
CONFIG_CC_OPTIMIZE_FOR_PERFORMANCE_O3=y
CONFIG_OPTIMIZE_INLINING=y
CONFIG_CC_OPTIMIZE_FOR_SIZE=n
CONFIG_LTO_CLANG_FULL=y  # If using Clang
```

---

## 🔧 Troubleshooting

### AVX-512 not detected after advanced unlock

**Step 1: Check status**:
```bash
sudo ./unlock_avx512_advanced.sh status
```

**Step 2: Verify /proc/cpuinfo**:
```bash
grep avx512 /proc/cpuinfo
# Should show: avx512f avx512dq avx512cd avx512bw avx512vl ...
```

**Step 3: If AVX-512 still not detected, microcode is blocking it**:
```bash
# Use the microcode fallback method
sudo ./unlock_avx512_advanced.sh microcode-fallback
sudo reboot
```

**Step 4: After reboot, verify again**:
```bash
./verify_avx512_advanced.sh
```

**Step 5: If still not working, check BIOS settings**:
- Boot into BIOS/UEFI (F2 during boot on Dell)
- Look for "Advanced CPU Configuration" or similar
- Check if AVX-512 is disabled in BIOS
- Some systems have "AVX-512 Fusing" option - ensure it's enabled

**Step 6: Check kernel support**:
```bash
# Ensure MSR module is available
modprobe msr
ls -la /dev/cpu/0/msr  # Should exist

# Install msr-tools if needed
sudo apt install msr-tools  # Debian/Ubuntu
sudo dnf install msr-tools  # Fedora/RHEL
```

### AVX-512 not detected after traditional unlock

**Check E-cores are actually disabled**:
```bash
sudo ./unlock_avx512.sh status
```

**Check /proc/cpuinfo**:
```bash
grep avx512 /proc/cpuinfo
# Should show: avx512f avx512dq avx512cd avx512bw avx512vl ...
```

**Try rebooting**:
```bash
sudo ./unlock_avx512.sh enable
sudo ./unlock_avx512.sh persistent
sudo reboot
```

---

### DSMIL driver not loading

**Check if driver exists**:
```bash
sudo find /lib/modules/$(uname -r) -name "dsmil*.ko"
```

**Manually load driver**:
```bash
sudo modprobe dsmil-84dev  # (legacy alias: dsmil-72dev)
# or
sudo insmod /path/to/dsmil-84dev.ko
#    (compatibility symlink /path/to/dsmil-72dev.ko is created automatically)
```

**Check driver status**:
```bash
lsmod | grep dsmil
dmesg | grep -i dsmil
```

**Note**: DSMIL driver is optional for AVX-512, but provides better Dell platform integration.

---

### Compilation fails with AVX-512 flags

**Check GCC version** (need 9.0+):
```bash
gcc --version
```

**Test basic AVX-512 support**:
```bash
echo 'int main(){return 0;}' | gcc -xc -mavx512f - -o /tmp/test
```

**Remove unsupported flags**:
```bash
# If some AVX-512 extensions fail, use minimal set:
export CFLAGS_AVX512_MINIMAL="-O3 -march=meteorlake -mavx512f -mavx512dq"
```

---

### Program crashes with "Illegal instruction"

**Ensure running on P-core**:
```bash
# Force execution on P-core (CPU 0-5)
taskset -c 0 ./myapp
```

**Check E-cores are disabled**:
```bash
cat /sys/devices/system/cpu/cpu6/online
# Should return: 0
```

**Verify AVX-512 at runtime**:
```bash
./verify_avx512.sh
```

---

### Kernel panic or boot issues

**Boot with E-cores enabled**:
```bash
# At GRUB, press 'e' and remove: systemd.unit=avx512-unlock.service
# Or boot into recovery mode and run:
sudo systemctl disable avx512-unlock.service
```

**Safe kernel build** (without AVX-512):
```bash
# Use regular Meteor Lake flags instead
make -j16 KCFLAGS="-O3 -march=meteorlake -mavx2"
```

---

## 📊 Performance Impact

### Workloads that benefit most (30-40% faster):
- ✅ Matrix multiplication
- ✅ Cryptographic hashing (SHA-256, SHA-512)
- ✅ Video encoding (x264, x265)
- ✅ Scientific computing (BLAS, LAPACK)
- ✅ Image processing
- ✅ AI inference (INT8/FP16 models)

### Workloads with minimal benefit (<10% faster):
- ❌ I/O bound tasks (file operations)
- ❌ Single-threaded code
- ❌ Small data sets (cache-friendly already)
- ❌ Non-vectorizable algorithms

### When to use AVX-512:
- Long-running compute jobs
- Batch processing
- Kernel compilation (15-25% faster)
- Development environment (compile often)

### When NOT to use AVX-512:
- General desktop use (need E-cores for responsiveness)
- Multi-tasking workloads
- Background services
- Gaming (needs all cores)

---

## 🔐 Security Considerations

### AVX-512 and side-channel attacks

AVX-512 can leak information via:
- **Power analysis**: 512-bit operations consume more power
- **Timing attacks**: Different execution time for AVX-512 vs AVX2
- **Cache effects**: Larger cache footprint

**Mitigation**:
```bash
# Use security-hardened flags
gcc $CFLAGS_AVX512_SECURE -o app app.c
```

### Recommended for security-critical code:
- Disable AVX-512 for crypto operations (use AES-NI instead)
- Use constant-time implementations
- Prefer AVX2 for sensitive workloads

---

## 📈 CPU Affinity

### Bind process to P-cores (required for AVX-512)

```bash
# Run on single P-core
taskset -c 0 ./myapp

# Run on all P-cores (0-5)
taskset -c 0-5 ./myapp

# Set affinity in code (C)
#define _GNU_SOURCE
#include <sched.h>

cpu_set_t set;
CPU_ZERO(&set);
CPU_SET(0, &set);  // Pin to CPU 0
sched_setaffinity(0, sizeof(set), &set);
```

### OpenMP with P-cores only

```bash
export GOMP_CPU_AFFINITY="0-5"
export OMP_NUM_THREADS="6"
export OMP_PROC_BIND="true"
export OMP_PLACES="cores"
```

---

## 🛠️ Advanced Usage

### Profile-Guided Optimization (PGO) with AVX-512

```bash
source ./avx512_compiler_flags.sh

# Stage 1: Generate profile
gcc $CFLAGS_AVX512 -fprofile-generate -o app_gen app.c
taskset -c 0 ./app_gen  # Run with typical workload

# Stage 2: Use profile
gcc $CFLAGS_AVX512 -fprofile-use -o app app.c
rm -f *.gcda
```

### Link-Time Optimization (LTO)

```bash
# Already included in CFLAGS_AVX512
# Force full LTO:
export CFLAGS_AVX512_LTO="$CFLAGS_AVX512 -flto=auto -fuse-linker-plugin"
export LDFLAGS_AVX512_LTO="$LDFLAGS_AVX512 -flto=auto"
```

---

## 📝 Notes

- **Hybrid architecture limitation**: This is a workaround, not a fix
- **Intel's decision**: AVX-512 disabled by design on Meteor Lake
- **Future Intel CPUs**: May have better hybrid + AVX-512 support
- **Alternative**: Use AVX2 + AVX-VNNI (no E-core disabling needed)

---

## 🔗 References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [GCC x86 Options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)
- [Linux kernel compilation](https://www.kernel.org/doc/html/latest/admin-guide/README.html)

---

## 📊 Method Comparison

| Feature | Advanced Method | Traditional Method |
|---------|-----------------|-------------------|
| **E-cores** | ✅ Remain active (10 cores) | ❌ Disabled (0 cores) |
| **Total cores** | 16 (6 P + 10 E) | 6 (P only) |
| **AVX-512 on P-cores** | ✅ Yes (with task pinning) | ✅ Yes (automatic) |
| **Multitasking** | ✅ Excellent | ⚠️ Limited |
| **Task pinning required** | ✅ Yes (critical) | ❌ No |
| **Setup complexity** | Medium | Simple |
| **Fallback options** | 2 (MSR + microcode) | 1 (E-core disable) |
| **DSMIL integration** | ✅ Yes | ❌ No |
| **Reboot required** | Only for fallback | No |
| **Best for** | Daily use + compute | Pure compute workloads |

**Recommendation**: Use **Advanced Method** for daily use where you need both multitasking and AVX-512. Use **Traditional Method** only if you exclusively run AVX-512 workloads and don't need E-cores.

---

## ✅ Command Summary

### Advanced Method
| Action | Command |
|--------|---------|
| Unlock AVX-512 (advanced) | `sudo ./unlock_avx512_advanced.sh enable` |
| Microcode fallback | `sudo ./unlock_avx512_advanced.sh microcode-fallback && sudo reboot` |
| Verify working | `./verify_avx512_advanced.sh` |
| Check status | `sudo ./unlock_avx512_advanced.sh status` |
| Run program on P-cores | `run-on-pcores ./myapp` or `taskset -c 0-5 ./myapp` |
| Restore microcode | `sudo ./unlock_avx512_advanced.sh restore-microcode && sudo reboot` |

### Traditional Method
| Action | Command |
|--------|---------|
| Unlock AVX-512 | `sudo ./unlock_avx512.sh enable` |
| Verify working | `./verify_avx512.sh` |
| Make persistent | `sudo ./unlock_avx512.sh persistent` |
| Lock AVX-512 | `sudo ./unlock_avx512.sh disable` |
| Check status | `sudo ./unlock_avx512.sh status` |

### Compilation (Both Methods)
| Action | Command |
|--------|---------|
| Source flags | `source ./avx512_compiler_flags.sh` |
| Compile app | `gcc $CFLAGS_AVX512 -o app app.c $LDFLAGS_AVX512` |
| Build kernel | `make -j6 KCFLAGS="$KCFLAGS_AVX512"` |
| Benchmark | `benchmark_avx512_vs_avx2` |

---

**Version**: 2.0 (Advanced P-core Task Pinning + Microcode Fallback)
**System**: Intel Core Ultra 7 165H (Meteor Lake)
**Dell Platform**: Latitude 5450 MIL-SPEC with DSMIL Integration
**Author**: KYBERLOCK Research Division

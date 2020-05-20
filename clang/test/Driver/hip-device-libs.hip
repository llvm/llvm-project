// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Test if oclc_daz_opt_on or if oclc_daz_opt_off is linked depending on
// expected denormal mode.

// Test subtarget with flushing on by default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD


// Test subtarget with flushing off by ddefault.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD


// Test last flag wins, not flushing
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD


// Test environment variable HIP_DEVICE_LIB_PATH

// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/hip_dev_lib \
// RUN:   %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM

// COM: {{"[^"]*clang[^"]*"}}
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}hip.amdgcn.bc"
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}ocml.amdgcn.bc"
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}ockl.amdgcn.bc"
// FLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_on.amdgcn.bc"
// NOFLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.amdgcn.bc"

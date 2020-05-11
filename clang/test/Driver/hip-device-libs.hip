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
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test subtarget with flushing off by ddefault.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test last flag wins, not flushing
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test environment variable HIP_DEVICE_LIB_PATH

// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/hip_dev_lib \
// RUN:   %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL

// ALL: {{"[^"]*clang[^"]*"}}
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}hip.amdgcn.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}ocml.amdgcn.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}ockl.amdgcn.bc"
// FLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_on.amdgcn.bc"
// NOFLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.amdgcn.bc"

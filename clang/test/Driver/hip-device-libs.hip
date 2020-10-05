// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Test if oclc_daz_opt_on or if oclc_daz_opt_off is linked depending on
// expected denormal mode.

// Test subtarget with flushing on by default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:  --cuda-gpu-arch=gfx803 \
// RUN:  --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test subtarget with flushing off by ddefault.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:  --cuda-gpu-arch=gfx900 \
// RUN:  --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test last flag wins, not flushing
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fcuda-flush-denormals-to-zero -fno-cuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-cuda-flush-denormals-to-zero -fcuda-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test --hip-device-lib-path flag
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 -nogpuinc \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test environment variable HIP_DEVICE_LIB_PATH
// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/rocm/amdgcn/bitcode \
// RUN:   %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL

// Test --gpu-instrument-lib
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   --gpu-instrument-lib=%S/Inputs/hip_multiple_inputs/instrument.bc \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,INST

// ALL-NOT: error:
// ALL: {{"[^"]*clang[^"]*"}}
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}hip.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}ocml.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}ockl.bc"

// FLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_on.bc"
// NOFLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.bc"

// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_unsafe_math_off.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_finite_only_off.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_correctly_rounded_sqrt_on.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_on.bc"
// ALL-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_isa_version_{{[0-9]+}}.bc"
// INST-SAME: "-mlink-builtin-bitcode" "{{.*}}instrument.bc"

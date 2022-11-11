// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Test if oclc_daz_opt_on or if oclc_daz_opt_off is linked depending on
// expected denormal mode.

// Test subtarget with flushing on by default.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:  --cuda-gpu-arch=gfx803 \
// RUN:  --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test subtarget with flushing off by ddefault.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:  --cuda-gpu-arch=gfx900 \
// RUN:  --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fgpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test explicit flag, opposite of target default.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-gpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-gpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// Test explicit flag, same as target default.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fgpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test last flag wins, not flushing
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fgpu-flush-denormals-to-zero -fno-gpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fgpu-flush-denormals-to-zero -fno-gpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,NOFLUSHD


// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -fno-gpu-flush-denormals-to-zero -fgpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 \
// RUN:   -fno-gpu-flush-denormals-to-zero -fgpu-flush-denormals-to-zero \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test --hip-device-lib-path flag
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 -nogpuinc \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode   \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,FLUSHD


// Test environment variable HIP_DEVICE_LIB_PATH
// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/rocm/amdgcn/bitcode \
// RUN:   %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL

// Test --gpu-instrument-lib
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   --gpu-instrument-lib=%S/Inputs/hip_multiple_inputs/instrument.bc \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ALL,INST

// Test -fast-math
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -ffast-math --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=FAST

// Test -ffinite-math-only
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -ffinite-math-only --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=FINITE

// Test -funsafe-math-optimizations
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -funsafe-math-optimizations --rocm-path=%S/Inputs/rocm \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=UNSAFE

// Test -fno-hip-fp32-correctly-rounded-divide-sqrt
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -fno-hip-fp32-correctly-rounded-divide-sqrt \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=DIVSQRT

// Test default code object version.
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ABI4

// Test default code object version with old device library without abi_version_400.bc
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode-no-abi-ver   \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=NOABI4

// Test -mcode-object-version=3
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -mcode-object-version=3 \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ABI4

// Test -mcode-object-version=4
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -mcode-object-version=4 \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ABI4

// Test -mcode-object-version=4 with old device library without abi_version_400.bc
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -mcode-object-version=4 \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode-no-abi-ver   \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=NOABI4

// Test -mcode-object-version=5
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -mcode-object-version=5 \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=ABI5

// Test -mcode-object-version=5 with old device library without abi_version_400.bc
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx900 \
// RUN:   -mcode-object-version=5 \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode-no-abi-ver   \
// RUN:   --rocm-path=%S/Inputs/rocm %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=NOABI5

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

// FAST: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.bc"
// FAST-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_unsafe_math_on.bc"
// FAST-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_finite_only_on.bc"
// FAST-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_correctly_rounded_sqrt_on.bc"

// FINITE: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.bc"
// FINITE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_unsafe_math_off.bc"
// FINITE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_finite_only_on.bc"
// FINITE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_correctly_rounded_sqrt_on.bc"

// UNSAFE: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.bc"
// UNSAFE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_unsafe_math_on.bc"
// UNSAFE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_finite_only_off.bc"
// UNSAFE-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_correctly_rounded_sqrt_on.bc"

// DIVSQRT: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.bc"
// DIVSQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_unsafe_math_off.bc"
// DIVSQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_finite_only_off.bc"
// DIVSQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_correctly_rounded_sqrt_off.bc"

// ABI4-NOT: error:
// ABI4: "-mlink-builtin-bitcode" "{{.*}}oclc_abi_version_400.bc"
// ABI5-NOT: error:
// ABI5: "-mlink-builtin-bitcode" "{{.*}}oclc_abi_version_500.bc"
// NOABI4-NOT: error:
// NOABI4-NOT: "-mlink-builtin-bitcode" "{{.*}}oclc_abi_version_400.bc"
// NOABI4-NOT: "-mlink-builtin-bitcode" "{{.*}}oclc_abi_version_500.bc"
// NOABI5: error: cannot find ROCm device libraryfor ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library

// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// case when amdgpu_arch returns nothing or fails
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%S/Inputs/amdgpu-arch/amdgpu_arch_fail %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: Cannot determine AMDGPU architecture. Consider passing it via -march

// case when amdgpu_arch returns multiple gpus but all are different
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%S/Inputs/amdgpu-arch/amdgpu_arch_different %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MULTIPLE-OUTPUT-ERROR
// MULTIPLE-OUTPUT-ERROR: error: Cannot determine AMDGPU architecture. Consider passing it via -march

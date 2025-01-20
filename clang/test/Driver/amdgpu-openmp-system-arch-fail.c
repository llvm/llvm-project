// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: rm -f %t/amdgpu_arch_fail %t/amdgpu_arch_different
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_fail %t/
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_different %t/
// RUN: echo '#!/bin/sh' > %t/amdgpu_arch_empty
// RUN: chmod +x %t/amdgpu_arch_fail
// RUN: chmod +x %t/amdgpu_arch_different
// RUN: chmod +x %t/amdgpu_arch_empty

// case when amdgpu_arch returns nothing or fails
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%t/amdgpu_arch_fail %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: cannot determine amdgcn architecture{{.*}}; consider passing it via '-march'

// case when amdgpu_arch does not return anything with successful execution
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%t/amdgpu_arch_empty %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// EMPTY-OUTPUT: error: cannot determine amdgcn architecture: No AMD GPU detected in the system; consider passing it via '-march'

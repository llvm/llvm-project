// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_fail %t/
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_gfx906 %t/
// RUN: echo '#!/bin/sh' > %t/amdgpu_arch_empty
// RUN: chmod +x %t/amdgpu_arch_fail
// RUN: chmod +x %t/amdgpu_arch_gfx906
// RUN: chmod +x %t/amdgpu_arch_empty

// case when amdgpu-arch returns nothing or fails
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_fail -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_fail -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: cannot determine amdgcn architecture{{.*}}; consider passing it via '--offload-arch'

// case when amdgpu-arch does not return anything with successful execution
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_empty -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_empty -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// EMPTY-OUTPUT: error: cannot determine amdgcn architecture: No AMD GPU detected in the system; consider passing it via '--offload-arch'

// case when amdgpu-arch returns a gfx906 GPU.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-GFX906
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-GFX906
// ARCH-GFX906: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"

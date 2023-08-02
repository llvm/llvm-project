// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_fail %t/
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_gfx906 %t/
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_fail %t/
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_sm_70 %t/
// RUN: echo '#!/bin/sh' > %t/amdgpu_arch_empty
// RUN: chmod +x %t/amdgpu_arch_fail
// RUN: chmod +x %t/amdgpu_arch_gfx906
// RUN: chmod +x %t/amdgpu_arch_empty
// RUN: echo '#!/bin/sh' > %t/nvptx_arch_empty
// RUN: chmod +x %t/nvptx_arch_fail
// RUN: chmod +x %t/nvptx_arch_sm_70
// RUN: chmod +x %t/nvptx_arch_empty

// case when nvptx-arch and amdgpu-arch return nothing or fails
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-arch=native \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_fail --amdgpu-arch-tool=%t/amdgpu_arch_fail %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-arch=native \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_empty --amdgpu-arch-tool=%t/amdgpu_arch_empty %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-arch= \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_fail --amdgpu-arch-tool=%t/amdgpu_arch_fail %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-arch= \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_empty --amdgpu-arch-tool=%t/amdgpu_arch_empty %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: failed to deduce triple for target architecture 'native'; specify the triple using '-fopenmp-targets' and '-Xopenmp-target' instead.

// case when amdgpu-arch succeeds.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-new-driver --offload-arch=native \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_fail --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-GFX906
// ARCH-GFX906: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"

// case when nvptx-arch succeeds.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-new-driver --offload-arch=native \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_sm_70 --amdgpu-arch-tool=%t/amdgpu_arch_fail %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-SM_70
// ARCH-SM_70: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"

// case when both nvptx-arch and amdgpu-arch succeed.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-new-driver --offload-arch=native \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_sm_70 --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-SM_70-GFX906
// ARCH-SM_70-GFX906: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"
// ARCH-SM_70-GFX906: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"

// case when both nvptx-arch and amdgpu-arch succeed with other archs.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp --offload-new-driver --offload-arch=native,sm_75,gfx1030 \
// RUN:     --nvptx-arch-tool=%t/nvptx_arch_sm_70 --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-MULTIPLE
// ARCH-MULTIPLE: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx1030"
// ARCH-MULTIPLE: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"
// ARCH-MULTIPLE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"
// ARCH-MULTIPLE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_75"

// case when 'nvptx-arch' returns nothing using `-fopenmp-targets=`.
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda --nvptx-arch-tool=%t/nvptx_arch_empty %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NVPTX
// NVPTX: error: cannot determine nvptx64 architecture: No NVIDIA GPU detected in the system; consider passing it via '-march'

// case when 'amdgpu-arch' returns nothing using `-fopenmp-targets=`.
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp \
// RUN:     -fopenmp-targets=amdgcn-amd-amdhsa --amdgpu-arch-tool=%t/amdgpu_arch_empty %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=AMDGPU
// AMDGPU: error: cannot determine amdgcn architecture: No AMD GPU detected in the system; consider passing it via '-march'

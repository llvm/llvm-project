// OpenMP offload to AMDGPU must add the ROCm include directories (where the HIP
// headers live) to both the host and device cc1 jobs.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
// RUN:   -march=gfx906 -nogpulib %s 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-internal-isystem" "{{.*}}/../../../include"
// CHECK: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-internal-isystem" "{{.*}}/../../../include"

// Test that --gpu-max-threads-per-block is not ignored by openmp.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   --offload-arch=gfx906 -nogpulib --gpu-max-threads-per-block=256 %s 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "--gpu-max-threads-per-block=256"

// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: clang-target-64-bits

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-targets=x86_64-unknown-unknown -o - | FileCheck --check-prefix=CHECK-45-VERSION %s
// CHECK-45-VERSION: #define _OPENMP 201511
// RUN: %clang %s -c -E -dM -fopenmp=libomp -nogpulib --offload-arch=sm_70 --offload-device-only -o - | FileCheck --check-prefix=CHECK-CUDA-ARCH %s
// CHECK-CUDA-ARCH: #define __CUDA_ARCH__ 700

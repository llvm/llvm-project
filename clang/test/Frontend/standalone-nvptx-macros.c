// REQUIRES: nvptx-registered-target

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_70 -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH %s
// CHECK-CUDA-ARCH: #define __CUDA_ARCH__ 700

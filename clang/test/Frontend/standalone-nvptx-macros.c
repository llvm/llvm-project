// REQUIRES: nvptx-registered-target

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_70 -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH %s
// CHECK-CUDA-ARCH: #define __CUDA_ARCH__ 700

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_100 -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH-100 %s
// CHECK-CUDA-ARCH-100: #define __CUDA_ARCH__ 1000

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_100 -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH-100-NEG %s
// CHECK-CUDA-ARCH-100-NEG-NOT: #define __CUDA_ARCH_FAMILY_SPECIFIC__
// CHECK-CUDA-ARCH-100-NEG-NOT: #define __CUDA_ARCH_FEAT_SM100_ALL
// CHECK-CUDA-ARCH-100-NEG-NOT: #define __CUDA_ARCH_SPECIFIC__

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_100f -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH-100F %s
// CHECK-CUDA-ARCH-100F-DAG: #define __CUDA_ARCH__ 1000
// CHECK-CUDA-ARCH-100F-DAG: #define __CUDA_ARCH_FAMILY_SPECIFIC__ 1000

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_100f -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH-100F-NEG %s
// CHECK-CUDA-ARCH-100F-NEG-NOT: #define __CUDA_ARCH_FEAT_SM100_ALL
// CHECK-CUDA-ARCH-100F-NEG-NOT: #define __CUDA_ARCH_SPECIFIC__

// RUN: %clang %s -c -E -dM --target=nvptx64-nvidia-cuda -march=sm_100a -o - | \
// RUN:   FileCheck --check-prefix=CHECK-CUDA-ARCH-100A %s
// CHECK-CUDA-ARCH-100A-DAG: #define __CUDA_ARCH__ 1000
// CHECK-CUDA-ARCH-100A-DAG: #define __CUDA_ARCH_FAMILY_SPECIFIC__ 1000
// CHECK-CUDA-ARCH-100A-DAG: #define __CUDA_ARCH_FEAT_SM100_ALL 1
// CHECK-CUDA-ARCH-100A-DAG: #define __CUDA_ARCH_SPECIFIC__ 1000

// Tests that during incremental HIP device compilation an inline __device__
// helper referenced by a kernel has its definition emitted into the device
// module (rather than left as an external declaration). In incremental mode
// device code objects are not linked together, so a referenced inline/internal
// __device__ function must be re-emitted into the referencing module.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fincremental-extensions -emit-llvm -o - %s | FileCheck %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ inline void test_device(int *value) { *value = 42; }
__global__ void test_kernel(int *value) { test_device(value); }

// CHECK: define {{.*}}@_Z11test_kernelPi
// CHECK: define {{.*}}@_Z11test_devicePi

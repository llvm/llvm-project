// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -fcuda-is-device \
// RUN: -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm %s \
// RUN: -o - | FileCheck %s

// Checks that builtins returning an i1 can be used as bool results implicitly.

#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z2shPv
// CHECK: call i1 @llvm.amdgcn.is.shared(ptr
__device__ bool sh(void *p) { return __builtin_amdgcn_is_shared(p); }

// CHECK-LABEL: @_Z3usePv
// CHECK: call i1 @llvm.amdgcn.is.shared(ptr
__device__ int use(void *p) {
  if (__builtin_amdgcn_is_shared(p))
    return 1;
  return 0;
}

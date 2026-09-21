// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -fcuda-is-device \
// RUN: -emit-cir %s -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -fcuda-is-device \
// RUN: -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
//
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm %s \
// RUN: -o - | FileCheck --check-prefix=LLVM %s

// Checks that builtins returning an i1 can be used as bool results implicitly.

#define __device__ __attribute__((device))

// CIR-LABEL: cir.func {{.*}} @_Z2shPv
// CIR: cir.call_llvm_intrinsic "amdgcn.is.shared" {{.*}} -> !cir.bool
// LLVM-LABEL: @_Z2shPv
// LLVM: call i1 @llvm.amdgcn.is.shared(ptr
__device__ bool sh(void *p) { return __builtin_amdgcn_is_shared(p); }

// CIR-LABEL: cir.func {{.*}} @_Z3usePv
// CIR: cir.call_llvm_intrinsic "amdgcn.is.shared" {{.*}} -> !cir.bool
// LLVM-LABEL: @_Z3usePv
// LLVM: call i1 @llvm.amdgcn.is.shared(ptr
__device__ int use(void *p) {
  if (__builtin_amdgcn_is_shared(p))
    return 1;
  return 0;
}

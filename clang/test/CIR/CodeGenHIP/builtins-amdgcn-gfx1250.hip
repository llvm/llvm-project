// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#define __device__ __attribute__((device))

// CIR-LABEL: @_Z13test_rcp_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_rcp_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
__device__ void test_rcp_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_rcp_bf16(a);
}

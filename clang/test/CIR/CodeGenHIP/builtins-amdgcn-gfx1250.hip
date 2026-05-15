#include "../CodeGenCUDA/Inputs/cuda.h"

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

// CIR-LABEL: @_Z14test_tanhf_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.tanh" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z14test_tanhf_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.tanh.f32(float %{{.*}})
__device__ void test_tanhf_f32(float* out, float a) {
  *out = __builtin_amdgcn_tanhf(a);
}

// CIR-LABEL: @_Z13test_tanh_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.tanh" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z13test_tanh_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.{{((amdgcn.){0,1})}}tanh.f16(half %{{.*}})
__device__ void test_tanh_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_tanhh(a);
}

// CIR-LABEL: @_Z14test_tanh_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.tanh" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z14test_tanh_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.tanh.bf16(bfloat %{{.*}})
__device__ void test_tanh_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_tanh_bf16(a);
}

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z13test_rcp_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_rcp_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
__device__ void test_rcp_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_rcp_bf16(a);
}

// CIR-LABEL: @_Z13test_rsq_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_rsq_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
__device__ void test_rsq_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_rsq_bf16(a);
}

// CIR-LABEL: @_Z14test_sqrt_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.sqrt" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z14test_sqrt_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
__device__ void test_sqrt_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_sqrt_bf16(a);
}

// CIR-LABEL: @_Z13test_sin_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.sin" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_sin_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.sin.bf16(bfloat %{{.*}})
__device__ void test_sin_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_sin_bf16(a);
}

// CIR-LABEL: @_Z13test_cos_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.cos" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_cos_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.cos.bf16(bfloat %{{.*}})
__device__ void test_cos_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_cos_bf16(a);
}

// CIR-LABEL: @_Z14test_exp2_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.exp2" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z14test_exp2_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.exp2.bf16(bfloat %{{.*}})
__device__ void test_exp2_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_exp2_bf16(a);
}

// CIR-LABEL: @_Z13test_log_bf16PDF16bDF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.log" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_log_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.log.bf16(bfloat %{{.*}})
__device__ void test_log_bf16(__bf16* out, __bf16 a) {
  *out = __builtin_amdgcn_log_bf16(a);
}

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
// LLVM: call{{.*}} half @llvm.amdgcn.tanh.f16(half %{{.*}})
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

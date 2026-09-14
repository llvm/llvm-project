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

// CIR-LABEL: @_Z19test_permlane_bcastPjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlane.bcast" {{.*}} : (!s32i, !s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z19test_permlane_bcastPjjjj
// LLVM: call{{.*}} i32 @llvm.amdgcn.permlane.bcast.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
__device__ void test_permlane_bcast(unsigned int* out, unsigned int src0,
                                    unsigned int src1, unsigned int src2) {
  *out = __builtin_amdgcn_permlane_bcast(src0, src1, src2);
}

// CIR-LABEL: @_Z16test_permlane_upPjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlane.up" {{.*}} : (!s32i, !s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z16test_permlane_upPjjjj
// LLVM: call{{.*}} i32 @llvm.amdgcn.permlane.up.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
__device__ void test_permlane_up(unsigned int* out, unsigned int src0,
                                 unsigned int src1, unsigned int src2) {
  *out = __builtin_amdgcn_permlane_up(src0, src1, src2);
}

// CIR-LABEL: @_Z18test_permlane_downPjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlane.down" {{.*}} : (!s32i, !s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z18test_permlane_downPjjjj
// LLVM: call{{.*}} i32 @llvm.amdgcn.permlane.down.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
__device__ void test_permlane_down(unsigned int* out, unsigned int src0,
                                   unsigned int src1, unsigned int src2) {
  *out = __builtin_amdgcn_permlane_down(src0, src1, src2);
}

// CIR-LABEL: @_Z17test_permlane_xorPjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlane.xor" {{.*}} : (!s32i, !s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z17test_permlane_xorPjjjj
// LLVM: call{{.*}} i32 @llvm.amdgcn.permlane.xor.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
__device__ void test_permlane_xor(unsigned int* out, unsigned int src0,
                                  unsigned int src1, unsigned int src2) {
  *out = __builtin_amdgcn_permlane_xor(src0, src1, src2);
}

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

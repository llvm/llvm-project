// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu8.02-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu8.02-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu8.02-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z18test_div_fixup_f16PDF16_DF16_DF16_DF16_
// CIR: ir.call_llvm_intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z18test_div_fixup_f16PDF16_DF16_DF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.div.fixup.f16(half %{{.+}}, half %{{.+}}, half %{{.+}})
__device__ void test_div_fixup_f16(_Float16* out, _Float16 a, _Float16 b, _Float16 c) {
  *out = __builtin_amdgcn_div_fixuph(a, b, c);
}

// CIR-LABEL: @_Z12test_rcp_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z12test_rcp_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.rcp.f16(half %{{.*}})
__device__ void test_rcp_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_rcph(a);
}

// CIR-LABEL: @_Z10test_rsq_hPDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z10test_rsq_hPDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.rsq.f16(half %{{.*}})
__device__ void test_rsq_h(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_rsqh(a);
}

// CIR-LABEL: @_Z13test_sqrt_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.sqrt" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z13test_sqrt_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.sqrt.f16(half %{{.*}})
__device__ void test_sqrt_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_sqrth(a);
}

// CIR-LABEL: @_Z12test_sin_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.sin" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z12test_sin_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.sin.f16(half %{{.*}})
__device__ void test_sin_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_sinh(a);
}

// CIR-LABEL: @_Z12test_cos_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.cos" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z12test_cos_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.cos.f16(half %{{.*}})
__device__ void test_cos_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_cosh(a);
}

// CIR-LABEL: @_Z19test_frexp_mant_f16PDF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.frexp.mant" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z19test_frexp_mant_f16PDF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.frexp.mant.f16(half %{{.*}})
__device__ void test_frexp_mant_f16(_Float16* out, _Float16 a) {
  *out = __builtin_amdgcn_frexp_manth(a);
}

// CIR-LABEL: @_Z14test_ldexp_f16PDF16_DF16_i
// CIR: [[TRUNC:%.*]] = cir.cast integral {{.*}} : !s32i -> !s16i
// CIR: cir.call_llvm_intrinsic "ldexp" {{.*}}, [[TRUNC]] : (!cir.f16, !s16i) -> !cir.f16
// LLVM: define{{.*}} void @_Z14test_ldexp_f16PDF16_DF16_i
// LLVM: [[TRUNC:%.*]] = trunc i32 {{.*}} to i16
// LLVM: call{{.*}} half @llvm.ldexp.f16.i16(half %{{.*}}, i16 [[TRUNC]])
__device__ void test_ldexp_f16(_Float16* out, _Float16 a, int b) {
    *out = __builtin_amdgcn_ldexph(a, b);
}

// CIR-LABEL: @_Z14test_class_f16PbDF16_i
// CIR: cir.call_llvm_intrinsic "amdgcn.class" {{.*}} : (!cir.f16, !s32i) -> !cir.bool
// LLVM: define{{.*}} void @_Z14test_class_f16PbDF16_i
// LLVM: call{{.*}} i1 @llvm.amdgcn.class.f16(half %{{.*}}, i32 %{{.*}})
__device__ void test_class_f16(bool* out, _Float16 a, int b) {
  *out = __builtin_amdgcn_classh(a, b);
}

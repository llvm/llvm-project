// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tonga -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx900 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tonga -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx900 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu tonga -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx900 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-llvm %s -o %t.ll
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

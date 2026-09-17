// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// __builtin_amdgcn_fmed3h requires gfx9-insts, so it lives here rather than in
// builtins-amdgcn.hip, mirroring builtins-amdgcn-gfx9.cl.

#define __device__ __attribute__((device))

// CIR-LABEL: @_Z14test_fmed3_f16PDF16_DF16_DF16_DF16_
// CIR: cir.call_llvm_intrinsic "amdgcn.fmed3" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @_Z14test_fmed3_f16PDF16_DF16_DF16_DF16_
// LLVM: call{{.*}} half @llvm.amdgcn.fmed3.f16(half %{{.+}}, half %{{.+}}, half %{{.+}})
__device__ void test_fmed3_f16(_Float16* out, _Float16 a, _Float16 b,
                               _Float16 c) {
  *out = __builtin_amdgcn_fmed3h(a, b, c);
}

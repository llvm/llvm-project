#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z18test_div_scale_f64PdPidd
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.double, !cir.double, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f64PdPidd
// LLVM: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
__device__ void test_div_scale_f64(double* out, int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z18test_div_scale_f32PfPbff
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f32PfPbff
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32(float* out, bool* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z27test_div_scale_f32_with_ptrPfPiPbff
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z27test_div_scale_f32_with_ptrPfPiPbff
// LLVM: {{.*}}call{{.*}} { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32_with_ptr(float* out, int* flagout, bool* flag, float a, float b)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

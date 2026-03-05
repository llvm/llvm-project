#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test logb/logbf builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z11test_logbfff
// CIR: cir.call @logbf({{.*}}) : (!cir.float) -> !cir.float
// LLVM: define{{.*}} float @_Z11test_logbfff(
// LLVM: call {{.*}}float @logbf(float %{{.*}})
// OGCG: define{{.*}} float @_Z11test_logbfff(
// OGCG: call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// OGCG: extractvalue { float, i32 } %{{.*}}, 1
// OGCG: add nsw i32 %{{.*}}, -1
// OGCG: sitofp i32 %{{.*}} to float
// OGCG: call {{.*}}float @llvm.fabs.f32(float %{{.*}})
// OGCG: fcmp {{.*}}one float %{{.*}}, 0x7FF0000000000000
// OGCG: select {{.*}}i1 %{{.*}}, float %{{.*}}, float %{{.*}}
// OGCG: fcmp {{.*}}oeq float %{{.*}}, 0.000000e+00
// OGCG: select {{.*}}i1 %{{.*}}, float 0xFFF0000000000000, float %{{.*}}
__device__ float test_logbff(float a) {
  return __builtin_logbf(a);
}

// CIR-LABEL: @_Z11test_logbddd
// CIR: cir.call @logb({{.*}}) : (!cir.double) -> !cir.double
// LLVM: define{{.*}} double @_Z11test_logbddd(
// LLVM: call {{.*}}double @logb(double %{{.*}})
// OGCG: define{{.*}} double @_Z11test_logbddd(
// OGCG: call { double, i32 } @llvm.frexp.f64.i32(double %{{.*}})
// OGCG: extractvalue { double, i32 } %{{.*}}, 1
// OGCG: add nsw i32 %{{.*}}, -1
// OGCG: sitofp i32 %{{.*}} to double
// OGCG: call {{.*}}double @llvm.fabs.f64(double %{{.*}})
// OGCG: fcmp {{.*}}one double %{{.*}}, 0x7FF0000000000000
// OGCG: select {{.*}}i1 %{{.*}}, double %{{.*}}, double %{{.*}}
// OGCG: fcmp {{.*}}oeq double %{{.*}}, 0.000000e+00
// OGCG: select {{.*}}i1 %{{.*}}, double 0xFFF0000000000000, double %{{.*}}
__device__ double test_logbdd(double a) {
  return __builtin_logb(a);
}

// CIR-LABEL: @_Z14test_scalbnffifi
// CIR: cir.call @scalbnf({{.*}}) : (!cir.float, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z14test_scalbnffifi(
// LLVM: call {{.*}}float @scalbnf(float %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} float @_Z14test_scalbnffifi(
// OGCG: call {{.*}}float @llvm.ldexp.f32.i32(float %{{.*}}, i32 %{{.*}})
__device__ float test_scalbnffi(float a, int b) {
  return __builtin_scalbnf(a, b);
}

// CIR-LABEL: @_Z14test_scalbnfdidi
// CIR: cir.call @scalbn({{.*}}) : (!cir.double, !s32i) -> !cir.double
// LLVM: define{{.*}} double @_Z14test_scalbnfdidi(
// LLVM: call {{.*}}double @scalbn(double %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} double @_Z14test_scalbnfdidi(
// OGCG: call {{.*}}double @llvm.ldexp.f64.i32(double %{{.*}}, i32 %{{.*}})
__device__ double test_scalbnfdi(double a, int b) {
  return __builtin_scalbn(a, b);
}

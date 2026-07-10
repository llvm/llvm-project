#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-LABEL: @_Z11test_logbfff
// CIR: cir.call_llvm_intrinsic "frexp" {{.*}} : (!cir.float) -> {{.*}}
// LLVM: define{{.*}} float @_Z11test_logbfff(
// LLVM: call { float, i32 } @llvm.frexp.f32.i32(float {{.*}})
__device__ float test_logbff(float a) {
  return __builtin_logbf(a);
}

// CIR-LABEL: @_Z11test_logbddd
// CIR: cir.call_llvm_intrinsic "frexp" {{.*}} : (!cir.double) -> {{.*}}
// LLVM: define{{.*}} double @_Z11test_logbddd(
// LLVM: call { double, i32 } @llvm.frexp.f64.i32(double {{.*}})
__device__ double test_logbdd(double a) {
  return __builtin_logb(a);
}

// CIR-LABEL: @_Z14test_scalbnffifi
// CIR: cir.call_llvm_intrinsic "ldexp" {{.*}} : (!cir.float, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z14test_scalbnffifi(
// LLVM: call {{.*}}float @llvm.ldexp.f32.i32(float {{.*}}, i32 {{.*}})
__device__ float test_scalbnffi(float a, int b) {
  return __builtin_scalbnf(a, b);
}

// CIR-LABEL: @_Z14test_scalbnfdidi
// CIR: cir.call_llvm_intrinsic "ldexp" {{.*}} : (!cir.double, !s32i) -> !cir.double
// LLVM: define{{.*}} double @_Z14test_scalbnfdidi(
// LLVM: call {{.*}}double @llvm.ldexp.f64.i32(double {{.*}}, i32 {{.*}})
__device__ double test_scalbnfdi(double a, int b) {
  return __builtin_scalbn(a, b);
}

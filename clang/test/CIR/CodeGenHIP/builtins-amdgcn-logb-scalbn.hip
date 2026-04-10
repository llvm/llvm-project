#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// CIR-LABEL: @_Z11test_logbfff
// CIR: cir.call @logbf({{.*}}){{.*}}: (!cir.float{{.*}}) -> !cir.float
// LLVM: define{{.*}} float @_Z11test_logbfff(
// LLVM: call {{.*}}float @logbf(float{{.*}}%{{.*}})
__device__ float test_logbff(float a) {
  return __builtin_logbf(a);
}

// CIR-LABEL: @_Z11test_logbddd
// CIR: cir.call @logb({{.*}}){{.*}}: (!cir.double{{.*}}) -> !cir.double
// LLVM: define{{.*}} double @_Z11test_logbddd(
// LLVM: call {{.*}}double @logb(double{{.*}}%{{.*}})
__device__ double test_logbdd(double a) {
  return __builtin_logb(a);
}

// CIR-LABEL: @_Z14test_scalbnffifi
// CIR: cir.call @scalbnf({{.*}}){{.*}}: (!cir.float{{.*}}, !s32i{{.*}}) -> !cir.float
// LLVM: define{{.*}} float @_Z14test_scalbnffifi(
// LLVM: call {{.*}}float @scalbnf(float{{.*}}%{{.*}}, i32{{.*}}%{{.*}})
__device__ float test_scalbnffi(float a, int b) {
  return __builtin_scalbnf(a, b);
}

// CIR-LABEL: @_Z14test_scalbnfdidi
// CIR: cir.call @scalbn({{.*}}){{.*}}: (!cir.double{{.*}}, !s32i{{.*}}) -> !cir.double
// LLVM: define{{.*}} double @_Z14test_scalbnfdidi(
// LLVM: call {{.*}}double @scalbn(double{{.*}}%{{.*}}, i32{{.*}}%{{.*}})
__device__ double test_scalbnfdi(double a, int b) {
  return __builtin_scalbn(a, b);
}

// RUN: %clang_cc1 %s -emit-llvm -o - -triple -amdgcn-amd-amdhsa \
// RUN:  -target-cpu gfx906 -fcuda-is-device -x hip \
// RUN:  | FileCheck --check-prefixes=COMMON,CRDIV %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple -amdgcn-amd-amdhsa \
// RUN:  -target-cpu gfx906 -fcuda-is-device -x hip \
// RUN:  -fno-hip-fp32-correctly-rounded-divide-sqrt \
// RUN:  | FileCheck --check-prefixes=COMMON,NCRDIV %s

#include "Inputs/cuda.h"

typedef __attribute__(( ext_vector_type(4) )) float float4;

// COMMON-LABEL: @_Z11spscalardiv
// COMMON: fdiv{{.*}},
// NCRDIV: !fpmath ![[MD:[0-9]+]]
// CRDIV-NOT: !fpmath
__device__ float spscalardiv(float a, float b) {
  return a / b;
}

// COMMON-LABEL: @_Z11spvectordiv
// COMMON: fdiv{{.*}},
// NCRDIV: !fpmath ![[MD]]
// CRDIV-NOT: !fpmath
__device__ float4 spvectordiv(float4 a, float4 b) {
  return a / b;
}

// COMMON-LABEL: @_Z11dpscalardiv
// COMMON-NOT: !fpmath
__device__ double dpscalardiv(double a, double b) {
  return a / b;
}

// NCRDIV: ![[MD]] = !{float 2.500000e+00}

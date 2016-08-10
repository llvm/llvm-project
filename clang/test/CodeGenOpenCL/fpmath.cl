// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck --check-prefix=CHECK --check-prefix=NODIVOPT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown -cl-fp32-correctly-rounded-divide-sqrt | FileCheck --check-prefix=CHECK --check-prefix=DIVOPT %s

typedef __attribute__(( ext_vector_type(4) )) float float4;

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv
  // CHECK: #[[ATTR:[0-9]+]]
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD:[0-9]+]]
  // DIVOPT-NOT: !fpmath ![[MD:[0-9]+]]
  return a / b;
}

float4 spvectordiv(float4 a, float4 b) {
  // CHECK: @spvectordiv
  // CHECK: #[[ATTR]]
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD]]
  // DIVOPT-NOT: !fpmath ![[MD]]
  return a / b;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double dpscalardiv(double a, double b) {
  // CHECK: @dpscalardiv
  // CHECK: #[[ATTR]]
  // CHECK-NOT: !fpmath
  return a / b;
}

// CHECK: attributes #[[ATTR]] = {
// NODIVOPT: "correctly-rounded-divide-sqrt-fp-math"="false"
// DIVOPT: "correctly-rounded-divide-sqrt-fp-math"="true"
// CHECK: }
// NODIVOPT: ![[MD]] = !{float 2.500000e+00}

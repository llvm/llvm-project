// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck --check-prefix=CHECK --check-prefix=NODIVOPT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown -cl-fp32-correctly-rounded-divide-sqrt | FileCheck --check-prefix=CHECK --check-prefix=DIVOPT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -DNOFP64 -cl-std=CL1.2 -triple r600-unknown-unknown -target-cpu r600 -pedantic | FileCheck --check-prefix=CHECK-FLT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -DFP64 -cl-std=CL1.2 -triple spir-unknown-unknown -pedantic | FileCheck --check-prefix=CHECK-DBL %s

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
  // CHECK: #[[ATTR2:[0-9]+]]
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD]]
  // DIVOPT-NOT: !fpmath ![[MD]]
  return a / b;
}

#if __OPENCL_C_VERSION__ >=120
void printf(constant char* fmt, ...);

void testdbllit(long *val) {
  // CHECK-FLT: float 2.000000e+01
  // CHECK-DBL: double 2.000000e+01
  printf("%f", 20.0);
}

#endif

#ifndef NOFP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double dpscalardiv(double a, double b) {
  // CHECK: @dpscalardiv
  // CHECK: #[[ATTR]]
  // CHECK-NOT: !fpmath
  return a / b;
}
#endif

// CHECK: attributes #[[ATTR]] = {
// NODIVOPT-SAME: "correctly-rounded-divide-sqrt-fp-math"="false"
// DIVOPT-SAME: "correctly-rounded-divide-sqrt-fp-math"="true"
// CHECK-SAME: }
// CHECK: attributes #[[ATTR2]] = {
// NODIVOPT-SAME: "correctly-rounded-divide-sqrt-fp-math"="false"
// DIVOPT-SAME: "correctly-rounded-divide-sqrt-fp-math"="true"
// CHECK-SAME: }
// NODIVOPT: ![[MD]] = !{float 2.500000e+00}

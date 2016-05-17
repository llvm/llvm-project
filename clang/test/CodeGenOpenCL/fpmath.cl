// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s

typedef __attribute__(( ext_vector_type(4) )) float float4;

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv
  // CHECK: fdiv{{.*}}, !fpmath ![[MD:[0-9]+]]
  return a / b;
}

float4 spvectordiv(float4 a, float4 b) {
  // CHECK: @spvectordiv
  // CHECK: fdiv{{.*}}, !fpmath ![[MD]]
  return a / b;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double dpscalardiv(double a, double b) {
  // CHECK: @dpscalardiv
  // CHECK-NOT: !fpmath
  return a / b;
}

// CHECK: ![[MD]] = !{float 2.500000e+00}

// RUN: %clang_cc1 -O3 -triple=aarch64-apple-ios -S -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

float fma_test1(float a, float b, float c) {
#pragma STDC FP_CONTRACT ON
// CHECK-LABEL: fma_test1:
// CHECK: fmadd
  float x = a * b + c;
  return x;
}

float fma_test2(float a, float b, float c) {
// CHECK-LABEL: fma_test2:
// CHECK: fmul
// CHECK: fadd
  float x = a * b + c;
  return x;
}

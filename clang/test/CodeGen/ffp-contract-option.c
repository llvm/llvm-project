// RUN: %clang_cc1 -O3 -ffp-contract=fast -triple=powerpc-apple-darwin10 -S -o - %s | FileCheck %s
// REQUIRES: ppc32-registered-target

float fma_test1(float a, float b, float c) {
// CHECK: fmadds
  float x = a * b;
  float y = x + c;
  return y;
}

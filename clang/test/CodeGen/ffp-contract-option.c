// RUN: %clang_cc1 -O3 -ffp-contract=fast -triple=aarch64-apple-darwin -S -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

float fma_test1(float a, float b, float c) {
// CHECK: fmadd
  float x = a * b;
  float y = x + c;
  return y;
}

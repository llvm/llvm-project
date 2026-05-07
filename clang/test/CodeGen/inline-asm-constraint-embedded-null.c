// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Regression test for issue173900.

// CHECK-LABEL: define {{.*}}void @f(
// CHECK: call void asm sideeffect "", "f,{{[^"]*}}"(double 0.000000e+00)
void f(void) { __asm__("" : : "f\0001"(0.0)); }

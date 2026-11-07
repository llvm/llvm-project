// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -o - %s | FileCheck %s


// GH204760

__complex__ double test_nonconstant(void) {
  __complex__ double b = __builtin_constant_p(b);
  return b;
}
// CHECK-LABEL: @test_nonconstant
// CHECK: ret { double, double } zeroinitializer

__complex__ double test_constant(void) {
  __complex__ double b = 1.0 + 2.0i;
  int c = __builtin_constant_p(b);
  __complex__ double r = c;
  return r;
}
// CHECK-LABEL: @test_constant
// CHECK: ret { double, double } { double 1.000000e+00, double 0.000000e+00 }

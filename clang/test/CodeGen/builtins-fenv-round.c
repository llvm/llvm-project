// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -Wno-unknown-pragmas | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter -emit-llvm %s -o - -Wno-unknown-pragmas | FileCheck %s

#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO

// CHECK-LABEL: define{{.*}} double @test_fdim(
// CHECK: ret double 1.000000e+00
double test_fdim(void) {
  return __builtin_fdim(__DBL_EPSILON__ / 2., -1.);
}

// CHECK-LABEL: define{{.*}} double @test_fma(
// CHECK: ret double 1.000000e+00
double test_fma(void) {
  return __builtin_fma(0.5, __DBL_EPSILON__, 1.0);
}

// CHECK-LABEL: define{{.*}} double @test_scalbn(
// CHECK: ret double 1.000000e+00
double test_scalbn(void) {
  return __builtin_scalbn(1.0 + __DBL_EPSILON__ / 2., 0);
}

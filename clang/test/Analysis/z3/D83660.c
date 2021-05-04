// RUN: rm -rf %t && mkdir %t
// RUN: %host_cxx -shared -fPIC %S/Inputs/MockZ3_solver_check.c -o %t/MockZ3_solver_check.so
//
// RUN: LD_PRELOAD="%t/MockZ3_solver_check.so"                                       \
// RUN: %clang_cc1 -analyze -analyzer-constraints=z3 -setup-static-analyzer          \
// RUN:   -analyzer-checker=core,debug.ExprInspection %s -verify 2>&1 | FileCheck %s
//
// REQUIRES: z3, asserts, shell, system-linux
//
// Works only with the z3 constraint manager.
// expected-no-diagnostics

// CHECK:      Z3_solver_check returns the real value: TRUE
// CHECK-NEXT: Z3_solver_check returns the real value: TRUE
// CHECK-NEXT: Z3_solver_check returns the real value: TRUE
// CHECK-NEXT: Z3_solver_check returns the real value: TRUE
// CHECK-NEXT: Z3_solver_check returns a mocked value: UNDEF

void D83660(int b) {
  if (b) {
  }
  (void)b; // no-crash
}

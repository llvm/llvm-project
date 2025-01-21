// RUN: rm -rf %t && mkdir %t
// RUN: %host_cxx -shared -fPIC \
// RUN:   %S/Inputs/MockZ3_solver_check.cpp \
// RUN:   -o %t/MockZ3_solver_check.so
//
// RUN: Z3_SOLVER_RESULTS="SAT,SAT,SAT,SAT,UNDEF" \
// RUN: LD_PRELOAD="%t/MockZ3_solver_check.so" \
// RUN: %clang_cc1 -analyze -analyzer-constraints=z3 -setup-static-analyzer \
// RUN:   -analyzer-checker=core %s -verify
//
// REQUIRES: z3, asserts, shell, system-linux
//
// Works only with the z3 constraint manager.
// expected-no-diagnostics

void D83660(int b) {
  if (b) {
  }
  (void)b; // no-crash
}

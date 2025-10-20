// RUN: env Z3_SOLVER_RESULTS="SAT,SAT,SAT,SAT,UNDEF" \
// RUN: LD_PRELOAD="%llvmshlibdir/MockZ3SolverCheck%pluginext" \
// RUN: %clang_analyze_cc1 -analyzer-constraints=z3 \
// RUN:   -analyzer-checker=core %s -verify
//
// REQUIRES: z3, z3-mock, asserts, shell, system-linux
//
// Works only with the z3 constraint manager.
// expected-no-diagnostics

void D83660(int b) {
  if (b) {
  }
  (void)b; // no-crash
}

// Check the default config.
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ConfigDumper 2>&1 \
// RUN: | FileCheck %s --match-full-lines
// CHECK: crosscheck-with-z3-max-attempts-per-query = 3

// DEFINE: %{mocked_clang} =                                          \
// DEFINE: env LD_PRELOAD="%llvmshlibdir/MockZ3SolverCheck%pluginext" \
// DEFINE: %clang_analyze_cc1 %s                                      \
// DEFINE:   -analyzer-config crosscheck-with-z3=true                 \
// DEFINE:   -analyzer-checker=core

// DEFINE: %{attempts} = -analyzer-config crosscheck-with-z3-max-attempts-per-query

// RUN: not %clang_analyze_cc1 %{attempts}=0 2>&1 | FileCheck %s --check-prefix=VERIFY-INVALID
// VERIFY-INVALID: invalid input for analyzer-config option 'crosscheck-with-z3-max-attempts-per-query', that expects a positive value

// RUN: env Z3_SOLVER_RESULTS="UNDEF"             %{mocked_clang} %{attempts}=1 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="UNSAT"             %{mocked_clang} %{attempts}=1 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="SAT"               %{mocked_clang} %{attempts}=1 -verify=accepted

// RUN: env Z3_SOLVER_RESULTS="UNDEF,UNDEF"       %{mocked_clang} %{attempts}=2 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="UNDEF,UNSAT"       %{mocked_clang} %{attempts}=2 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="UNDEF,SAT"         %{mocked_clang} %{attempts}=2 -verify=accepted

// RUN: env Z3_SOLVER_RESULTS="UNDEF,UNDEF,UNDEF" %{mocked_clang} %{attempts}=3 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="UNDEF,UNDEF,UNSAT" %{mocked_clang} %{attempts}=3 -verify=refuted
// RUN: env Z3_SOLVER_RESULTS="UNDEF,UNDEF,SAT"   %{mocked_clang} %{attempts}=3 -verify=accepted


// REQUIRES: z3, z3-mock, asserts, shell, system-linux

// refuted-no-diagnostics

int div_by_zero_test(int b) {
  if (b) {}
  return 100 / b; // accepted-warning {{Division by zero}}
}

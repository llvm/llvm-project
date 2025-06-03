// Check the default config.
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ConfigDumper 2>&1 \
// RUN: | FileCheck %s --match-full-lines
// CHECK: crosscheck-with-z3-max-attempts-per-query = 3

// RUN: rm -rf %t && mkdir %t
// RUN: %host_cxx -shared -fPIC                           \
// RUN:   %S/z3/Inputs/MockZ3_solver_check.cpp            \
// RUN:   -o %t/MockZ3_solver_check.so

// DEFINE: %{mocked_clang} =                              \
// DEFINE: LD_PRELOAD="%t/MockZ3_solver_check.so"         \
// DEFINE: %clang_cc1 %s -analyze -setup-static-analyzer  \
// DEFINE:   -analyzer-config crosscheck-with-z3=true     \
// DEFINE:   -analyzer-checker=core

// DEFINE: %{attempts} = -analyzer-config crosscheck-with-z3-max-attempts-per-query

// RUN: not %clang_analyze_cc1 %{attempts}=0 2>&1 | FileCheck %s --check-prefix=VERIFY-INVALID
// VERIFY-INVALID: invalid input for analyzer-config option 'crosscheck-with-z3-max-attempts-per-query', that expects a positive value

// RUN: Z3_SOLVER_RESULTS="UNDEF"             %{mocked_clang} %{attempts}=1 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="UNSAT"             %{mocked_clang} %{attempts}=1 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="SAT"               %{mocked_clang} %{attempts}=1 -verify=accepted

// RUN: Z3_SOLVER_RESULTS="UNDEF,UNDEF"       %{mocked_clang} %{attempts}=2 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="UNDEF,UNSAT"       %{mocked_clang} %{attempts}=2 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="UNDEF,SAT"         %{mocked_clang} %{attempts}=2 -verify=accepted

// RUN: Z3_SOLVER_RESULTS="UNDEF,UNDEF,UNDEF" %{mocked_clang} %{attempts}=3 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="UNDEF,UNDEF,UNSAT" %{mocked_clang} %{attempts}=3 -verify=refuted
// RUN: Z3_SOLVER_RESULTS="UNDEF,UNDEF,SAT"   %{mocked_clang} %{attempts}=3 -verify=accepted


// REQUIRES: z3, asserts, shell, system-linux

// refuted-no-diagnostics

int div_by_zero_test(int b) {
  if (b) {}
  return 100 / b; // accepted-warning {{Division by zero}}
}

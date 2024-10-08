// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s  \
// RUN:   -analyzer-config crosscheck-with-z3=true \
// RUN:   -analyzer-stats 2>&1 | FileCheck %s

// REQUIRES: z3

// expected-error@1 {{Z3 refutation rate:1/2}}

int accepting(int n) {
  if (n == 4) {
    n = n / (n-4); // expected-warning {{Division by zero}}
  }
  return n;
}

int rejecting(int n, int x) {
  // Let's make the path infeasible.
  if (2 < x && x < 5 && x*x == x*x*x) {
    // Have the same condition as in 'accepting'.
    if (n == 4) {
      n = x / (n-4); // no-warning: refuted
    }
  }
  return n;
}

// CHECK:       1 BugReporter         - Number of times all reports of an equivalence class was refuted
// CHECK-NEXT:  1 BugReporter         - Number of reports passed Z3
// CHECK-NEXT:  1 BugReporter         - Number of reports refuted by Z3

// CHECK:       1 Z3CrosscheckVisitor - Number of Z3 queries accepting a report
// CHECK-NEXT:  1 Z3CrosscheckVisitor - Number of Z3 queries rejecting a report
// CHECK-NEXT:  2 Z3CrosscheckVisitor - Number of Z3 queries done

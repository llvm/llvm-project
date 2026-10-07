// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-constraints=unsupported-z3 -verify %s
// REQUIRES: z3

void clang_analyzer_eval(int);

void indirect_constraints(int a, int b, int c) {
  if (a != b && b == c && c == 42) {
    clang_analyzer_eval(b == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(a != 42); // expected-warning{{TRUE}}
  }
}

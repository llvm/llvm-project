// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -w \
// RUN:   -analyzer-config crosscheck-with-z3=true -verify %s
// REQUIRES: z3

// Previously these tests were crashing because the SMTConv layer did not
// comprehend the _BitInt types.

void clang_analyzer_warnIfReached();

void c(int b, _BitInt(35) a) {
  int d;
  if (a)
    b = d; // expected-warning{{Assigned value is uninitialized [core.uninitialized.Assign]}}
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f(int *d, _BitInt(3) e) {
  int g;
  d = &g;
  e ?: 0;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

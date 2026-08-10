// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// Here we test whether the SValBuilder is capable to simplify existing
// SVals based on a newly added constraints when evaluating a BinOp.

void clang_analyzer_eval(bool);

void test_evalBinOp_simplifies_lhs(int y) {
  int x = y / 77;
  if (y != 77)
    return;

  // Below `x` is the LHS being simplified.
  clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}
  (void)(x * y);
}

void test_evalBinOp_simplifies_rhs(int y) {
  int x = y / 77;
  if (y != 77)
    return;

  // Below `x` is the RHS being simplified.
  clang_analyzer_eval(1 == x); // expected-warning{{TRUE}}
  (void)(x * y);
}

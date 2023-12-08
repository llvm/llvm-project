// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// Here we test whether the SValBuilder is capable to simplify existing
// IntSym expressions based on a newly added constraint on the sub-expression.

void clang_analyzer_eval(bool);

void test_SValBuilder_simplifies_IntSym(int x, int y) {
  // Most IntSym BinOps are transformed to SymInt in SimpleSValBuilder.
  // Division is one exception.
  x = 77 / y;
  if (y != 1)
    return;
  clang_analyzer_eval(x == 77); // expected-warning{{TRUE}}
  (void)(x * y);
}

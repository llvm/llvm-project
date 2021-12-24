// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// In this test we check whether the solver's symbol simplification mechanism
// is capable of re-assuming simiplified symbols.

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

void test_reassume_false_range(int x, int y) {
  if (x + y != 0) // x + y == 0
    return;
  if (x != 1)     // x == 1
    return;
  clang_analyzer_eval(y == -1); // expected-warning{{TRUE}}
}

void test_reassume_true_range(int x, int y) {
  if (x + y != 1) // x + y == 1
    return;
  if (x != 1)     // x == 1
    return;
  clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
}

void test_reassume_inclusive_range(int x, int y) {
  if (x + y < 0 || x + y > 1) // x + y: [0, 1]
    return;
  if (x != 1)                 // x == 1
    return;
                              // y: [-1, 0]
  clang_analyzer_eval(y > 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(y < -1);// expected-warning{{FALSE}}
}

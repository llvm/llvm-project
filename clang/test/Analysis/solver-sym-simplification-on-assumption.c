// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_eval(int);

void test_derived_sym_simplification_on_assume(int s0, int s1) {
  int elem = s0 + s1 + 1;
  if (elem-- == 0) // elem = s0 + s1
    return;

  if (elem-- == 0) // elem = s0 + s1 - 1
    return;

  if (s0 < 1) // s0: [1, 2147483647]
    return;
  if (s1 < 1) // s0: [1, 2147483647]
    return;

  if (elem-- == 0) // elem = s0 + s1 - 2
    return;

  if (s0 > 1) // s0: [-2147483648, 0] U [1, 2147483647] => s0 = 0
    return;

  if (s1 > 1) // s1: [-2147483648, 0] U [1, 2147483647] => s1 = 0
    return;

  // elem = s0 + s1 - 2 should be 0
  clang_analyzer_eval(elem); // expected-warning{{FALSE}}
}

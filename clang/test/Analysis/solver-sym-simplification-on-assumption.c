// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

void clang_analyzer_eval(int);
void clang_analyzer_value(int);

void test_derived_sym_simplification_on_assume(int s0, int s1) {
  int elem = s0 + s1 + 1;
  if (elem-- == 0)
    return;

  if (elem-- == 0)
    return;

  if (s0 < 1)
    return;
  clang_analyzer_value(s0); // expected-warning{{[1, 2147483647]}}

  if (s1 < 1)
    return;
  clang_analyzer_value(s1); // expected-warning{{[1, 2147483647]}}

  clang_analyzer_eval(elem); // expected-warning{{UNKNOWN}}
  if (elem-- == 0)
    return;

  if (s0 > 1)
    return;
  clang_analyzer_eval(s0 == 1); // expected-warning{{TRUE}}

  if (s1 > 1)
    return;
  clang_analyzer_eval(s1 == 1); // expected-warning{{TRUE}}

  clang_analyzer_eval(elem); // expected-warning{{FALSE}}
}

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-constraints=unsupported-z3 -verify %s
// REQUIRES: z3

void clang_analyzer_eval(int);

void unary_not_logical_result(int x, int y) {
  clang_analyzer_eval(~(x && y) != 0); // expected-warning{{TRUE}}
}

void unary_minus_logical_result(int x, int y) {
  clang_analyzer_eval(-(x && y) <= 0); // expected-warning{{TRUE}}
}

void wide_bitint_logical_truth_value(unsigned _BitInt(63) x) {
  clang_analyzer_eval((1 && x) == (x != 0)); // expected-warning{{TRUE}}
}

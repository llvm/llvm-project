// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);

void top(int b, int c) {
  if (c >= b) {
    clang_analyzer_eval(c >= b); // expected-warning{{TRUE}}
    clang_analyzer_eval(b <= c); // expected-warning{{TRUE}}
    clang_analyzer_eval((b - 0) <= (c + 0)); // expected-warning{{TRUE}}
    clang_analyzer_eval(b + 0 <= c + 0); // expected-warning{{TRUE}}
  }
}

void comparisons_imply_size(unsigned long lhs, unsigned long rhs) {
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{UNKNOWN}}

  if (lhs > rhs) {
    clang_analyzer_eval(rhs == lhs); // expected-warning{{FALSE}}
    clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
    clang_analyzer_eval(lhs != rhs); // expected-warning{{TRUE}}
    clang_analyzer_eval(lhs - rhs == 0); // expected-warning{{FALSE}}
    clang_analyzer_eval(rhs - lhs == 0); // expected-warning{{FALSE}}
  }
}

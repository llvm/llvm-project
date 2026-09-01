// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void test_zero_initialized_new_array() {
  int *p = new int[10]{};
  clang_analyzer_eval(*p == 0); // expected-warning{{TRUE}}
  delete[] p;
}

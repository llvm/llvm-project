// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void recursive_func(int *p, int count) {
  if (count == 0) return;
  int x = *p;
  recursive_func(p, count - 1);
  clang_analyzer_eval(*p == x); // expected-warning{{TRUE}}
}

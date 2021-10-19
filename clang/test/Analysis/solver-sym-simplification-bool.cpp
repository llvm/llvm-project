// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core \
// RUN: -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_dump(bool);

void foo(int &x) {
  int *p = &x; // 'p' is the same SVal as 'x'
  bool b = p;
  clang_analyzer_dump(b); // expected-warning {{1 U1b}}
}

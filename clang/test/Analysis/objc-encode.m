// RUN: %clang_cc1 -analyze -analyzer-checker=debug.ExprInspection -verify %s
// expected-no-diagnostics

void clang_analyzer_eval(int);

// rdar://problem/34831581: Used to crash.
void foo(void) {
  char buf1[] = @encode(int **);
}

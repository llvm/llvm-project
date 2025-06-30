// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s
// expected-no-diagnostics

void clang_analyzer_eval(int);

// Used to crash.
void foo(void) {
  char buf1[] = @encode(int **);
}

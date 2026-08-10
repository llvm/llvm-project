// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

// Here, we test that symbol simplification in the solver does not produce any
// crashes.
// https://bugs.llvm.org/show_bug.cgi?id=51109

// expected-no-diagnostics

int a, b, c, d;
void f(void) {
  a = -1;
  d = b * a;
  a = d / c;
  if (a < 7 / b)
    return;
  if (d *a / c < 7 / b)
    return;
  if (b == 1 && c == -1)
    return;
}

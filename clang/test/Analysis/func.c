// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core,debug.ExprInspection -analyzer-store=region -verify %s

void clang_analyzer_eval(int);

void f(void) {
  void (*p)(void);
  p = f;
  p = &f;
  p();
  (*p)();
}

void g(void (*fp)(void));

void f2() {
  g(f);
}

void f3(void (*f)(void), void (*g)(void)) {
  clang_analyzer_eval(!f); // expected-warning{{UNKNOWN}}
  f();
  clang_analyzer_eval(!f); // expected-warning{{FALSE}}

  clang_analyzer_eval(!g); // expected-warning{{UNKNOWN}}
  (*g)();
  clang_analyzer_eval(!g); // expected-warning{{FALSE}}
}

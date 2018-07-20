// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core.builtin,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void initbug() {
  const union { float a; } u = {};
  (void)u.a; // no-crash
}

int const parr[2] = {1};
void constarr() {
  int i = 2;
  clang_analyzer_eval(parr[i]); // expected-warning{{UNDEFINED}}
  i = 1;
  clang_analyzer_eval(parr[i] == 0); // expected-warning{{TRUE}}
  i = -1;
  clang_analyzer_eval(parr[i]); // expected-warning{{UNDEFINED}}
}

struct SM {
  int a;
  int b;
};
const struct SM sm = {.a = 1};
void multinit() {
  clang_analyzer_eval(sm.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(sm.b == 0); // expected-warning{{TRUE}}
}

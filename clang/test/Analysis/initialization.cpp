// RUN: %clang_cc1 -std=c++14 -triple i386-apple-darwin10 -analyze -analyzer-checker=core.builtin,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

struct S {
  int a = 3;
};
S const sarr[2] = {};
void definit() {
  int i = 1;
  // FIXME: Should recognize that it is 3.
  clang_analyzer_eval(sarr[i].a); // expected-warning{{UNKNOWN}}
}

int const arr[2][2] = {};
void arr2init() {
  int i = 1;
  // FIXME: Should recognize that it is 0.
  clang_analyzer_eval(arr[i][0]); // expected-warning{{UNKNOWN}}
}

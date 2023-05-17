// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

struct s { int a; };
int foo() {
  auto [a] = s{1};
  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
} // expected-warning{{non-void function does not return a value}}

struct s2 {
  int &x;
};

int *foo2(s2 in) {
  auto [a] = in;
  return &a;
}

void bar() {
  int i = 1;
  s2 a{i};

  auto *x = foo2(a);

  clang_analyzer_eval(*x == i); // expected-warning{{TRUE}}

  *x = 2;

  clang_analyzer_eval(*x == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(i == 2);  // expected-warning{{TRUE}}
}

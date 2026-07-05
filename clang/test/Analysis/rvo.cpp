// RUN: %clang_analyze_cc1 -analyzer-checker core,cplusplus -std=c++14 \
// RUN:                    -analyzer-checker debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

struct A {
  int x;
};

A getA();

struct B {
  int *p;
  A a;

  B(int *p) : p(p), a(getA()) {}
};

void foo() {
  B b1(nullptr);
  clang_analyzer_eval(b1.p == nullptr); // expected-warning{{TRUE}}
  B b2(new int); // No leak yet!
  clang_analyzer_eval(b2.p == nullptr); // expected-warning{{FALSE}}
  // expected-warning@-1{{Potential leak of memory pointed to by 'b2.p'}}
}

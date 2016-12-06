// RUN: %clang_cc1 -Wno-unused -std=c++11 -analyze -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

namespace pr17001_call_wrong_destructor {
bool x;
struct A {
  int *a;
  A() {}
  ~A() {}
};
struct B : public A {
  B() {}
  ~B() { x = true; }
};

void f() {
  {
    const A &a = B();
  }
  clang_analyzer_eval(x); // expected-warning{{TRUE}}
}
} // end namespace pr17001_call_wrong_destructor

namespace pr19539_crash_on_destroying_an_integer {
struct A {
  int i;
  int j[2];
  A() : i(1) {
    j[0] = 2;
    j[1] = 3;
  }
  ~A() {}
};

void f() {
  const int &x = A().i; // no-crash
  const int &y = A().j[1]; // no-crash
  const int &z = (A().j[1], A().j[0]); // no-crash

  // FIXME: All of these should be TRUE, but constructors aren't inlined.
  clang_analyzer_eval(x == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z == 2); // expected-warning{{UNKNOWN}}
}
} // end namespace pr19539_crash_on_destroying_an_integer

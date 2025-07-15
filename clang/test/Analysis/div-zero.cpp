// RUN: %clang_analyze_cc1 -analyzer-checker=core.DivideZero -std=c++20 -verify %s

namespace GH10616 {
int foo(int qX) {
  int a, c, d;

  d = (qX - 1);
  while (d != 0) {
    d = c - (c / d) * d;
  }

  return (a % (qX - 1)); // expected-warning {{Division by zero}}
}
} // namespace GH10616

namespace GH148875 {
struct A {
  int x;
  A(int v) : x(v) {}
};

struct B {
  int x;
  B() : x(0) {}
};

struct C {
  int x, y;
  C(int a, int b) : x(a), y(b) {}
};

int t1() {
  A a(42);
  return 1 / (a.x - 42); // expected-warning {{Division by zero}}
}

int t2() {
  B b;
  return 1 / b.x; // expected-warning {{Division by zero}}
}

int t3() {
  C c1(1, -1);
  return 1 / (c1.x + c1.y); // expected-warning {{Division by zero}}
}

int t4() {
  C c2(0, 0);
  return 1 / (c2.x + c2.y); // expected-warning {{Division by zero}}
}
} // namespace GH148875

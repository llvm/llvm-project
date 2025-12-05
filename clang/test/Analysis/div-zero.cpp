// RUN: %clang_analyze_cc1 -analyzer-checker=core.DivideZero -verify %s

int fooPR10616 (int qX ) {
  int a, c, d;

  d = (qX-1);
  while ( d != 0 ) {
    d = c - (c/d) * d;
  }

  return (a % (qX-1)); // expected-warning {{Division by zero}}

}

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

struct D {
  int x;
};

struct E {
  D d;
  E(int a) : d{a} {}
};

struct F {
  int x;
};

int t1() {
  A a{42};
  return 1 / (a.x - 42); // expected-warning {{Division by zero}}
}

int t2() {
  B b{};
  return 1 / b.x; // expected-warning {{Division by zero}}
}

int t3() {
  C c1{1, -1};
  return 1 / (c1.x + c1.y); // expected-warning {{Division by zero}}
}

int t4() {
  C c2{0, 0};
  return 1 / (c2.x + c2.y); // expected-warning {{Division by zero}}
}

int t5() {
  E e{32};
  return 1 / (e.d.x - 32); // expected-warning {{Division by zero}}
}

int t6() {
  F f{32};
  return 1 / (f.x - 32); // expected-warning {{Division by zero}}
}
}

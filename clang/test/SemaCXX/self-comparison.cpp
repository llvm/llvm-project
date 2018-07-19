// RUN: %clang_cc1 -fsyntax-only -verify %s

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always evaluates to true}}
}

struct X {
  bool operator==(const X &x);
};

struct A {
  int x;
  X x2;
  int a[3];
  int b[3];
  bool f() { return x == x; } // expected-warning {{self-comparison always evaluates to true}}
  bool g() { return x2 == x2; } // no-warning
  bool h() { return a == b; } // expected-warning {{array comparison always evaluates to false}}
  bool i() {
    int c[3];
    return a == c; // expected-warning {{array comparison always evaluates to false}}
  }
};

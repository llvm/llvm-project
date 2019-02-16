// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wexit-time-destructors %s -verify

namespace test1 {
  struct A { ~A(); };
  A a; // expected-warning {{declaration requires an exit-time destructor}}
  A b[10]; // expected-warning {{declaration requires an exit-time destructor}}
  A c[10][10]; // expected-warning {{declaration requires an exit-time destructor}}

  A &d = a;
  A &e = b[5];
  A &f = c[5][7];
}

namespace test2 {
void f() {
  struct A { ~A() { } };
  
  static A a; // expected-warning {{declaration requires an exit-time destructor}}
  static A b[10]; // expected-warning {{declaration requires an exit-time destructor}}
  static A c[10][10]; // expected-warning {{declaration requires an exit-time destructor}}

  static A &d = a;
  static A &e = b[5];
  static A &f = c[5][7];
}
}

namespace test3 {
  struct A { ~A() = default; };
  A a;

  struct B { ~B(); };
  struct C : B { ~C() = default; };
  C c; // expected-warning {{exit-time destructor}}

  class D {
    friend struct E;
    ~D() = default;
  };
  struct E : D {
    D d;
    ~E() = default;
  };
  E e;
}

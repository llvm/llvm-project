// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

namespace ExplicitConv {
  struct X { }; // expected-note 2{{candidate constructor}}

  struct Y {
    explicit operator X() const; // expected-note {{not a candidate}}
  };

  void test(const Y& y) {
    X x(static_cast<X>(y));
    X x2((X)y);
    X x3 = y; // expected-error{{no viable conversion from 'const Y' to 'X'}}
  }
}

namespace DR899 {
  struct C { }; // expected-note 2 {{candidate constructor}}

  struct A {
    explicit operator int() const; // expected-note {{not a candidate}}
    explicit operator C() const; // expected-note {{not a candidate}}
  };

  struct B {
    int i;
    B(const A& a): i(a) { }
  };

  int main() {
    A a;
    int i = a; // expected-error{{no viable conversion}}
    int j(a);
    C c = a; // expected-error{{no viable conversion}}
    C c2(a);
  }
}

// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// C++0x N2914.

struct B {
  void f(char);
  void g(char);
  enum E { e };
  union { int x; };
};

class C {
  int g();
};

class D2 : public B {
  using B::f;
  using B::e;
  using B::x;
  using C::g; // expected-error{{using declaration refers into 'C::', which is not a base class of 'D2'}}
};

namespace test1 {
  struct Base {
    int foo();
  };

  struct Unrelated {
    int foo();
  };

  struct Subclass : Base {
  };

  namespace InnerNS {
    int foo();
  }

  // We should be able to diagnose these without instantiation.
  template <class T> struct C : Base {
    using InnerNS::foo; // expected-error {{not a class}}
    using Base::bar; // expected-error {{no member named 'bar'}}
    using Unrelated::foo; // expected-error {{not a base class}}
    using C::foo; // expected-error {{refers to its own class}}
    using Subclass::foo; // expected-error {{not a base class}}
  };
}

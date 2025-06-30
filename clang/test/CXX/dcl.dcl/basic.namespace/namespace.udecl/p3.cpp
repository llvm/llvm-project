// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

struct B {
  void f(char);
  void g(char);
  enum E { e };
  union { int x; };

  enum class EC { ec }; // expected-warning 0-1 {{C++11}}

  void f2(char);
  void g2(char);
  enum E2 { e2 };
  union { int x2; };
};

class C {
public:
  int g();
};

struct D : B {};

class D2 : public B {
  using B::f;
  using B::E;
  using B::e;
  using B::x;
  using C::g; // expected-error{{using declaration refers into 'C', which is not a base class of 'D2'}}

  using D::f2; // expected-error {{using declaration refers into 'D', which is not a base class of 'D2'}}
  using D::E2; // expected-error {{using declaration refers into 'D', which is not a base class of 'D2'}}
  using D::e2; // expected-error {{using declaration refers into 'D', which is not a base class of 'D2'}}
  using D::x2; // expected-error {{using declaration refers into 'D', which is not a base class of 'D2'}}

  using B::EC;
  using B::EC::ec; // expected-warning {{a C++20 extension}} expected-warning 0-1 {{C++11}}
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

  struct B : Base {
  };

  // We should be able to diagnose these without instantiation.
  template <class T> struct C : Base {
    using InnerNS::foo; // expected-error {{not a class}}
    using Base::bar; // expected-error {{no member named 'bar'}}
    using Unrelated::foo; // expected-error {{not a base class}}

    using C::foo; // expected-error {{refers to its own class}}
    using Subclass::foo; // expected-error {{not a base class}}
  };
}

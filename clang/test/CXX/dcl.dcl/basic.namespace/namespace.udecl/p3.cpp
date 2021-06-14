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
  using C::g; // expected-error{{using declaration refers into 'C::', which is not a base class of 'D2'}}

  // These are valid in C++98 but not in C++11.
  using D::f2;
  using D::E2;
  using D::e2;
  using D::x2;
#if __cplusplus >= 201103L
  // expected-error@-5 {{using declaration refers into 'D::', which is not a base class of 'D2'}}
  // expected-error@-5 {{using declaration refers into 'D::', which is not a base class of 'D2'}}
  // expected-error@-5 {{using declaration refers into 'D::', which is not a base class of 'D2'}}
  // expected-error@-5 {{using declaration refers into 'D::', which is not a base class of 'D2'}}
#endif

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

    // In C++98, it's hard to see that these are invalid, because indirect
    // references to base class members are permitted.
    using C::foo;
    using Subclass::foo;
#if __cplusplus >= 201103L
    // expected-error@-3 {{refers to its own class}}
    // expected-error@-3 {{not a base class}}
#endif
  };
}

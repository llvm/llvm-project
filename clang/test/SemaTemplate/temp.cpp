// RUN: %clang_cc1 -fsyntax-only -verify %s


namespace test0 {
  // p3
  template<typename T> int foo(T), bar(T, T); // expected-error{{single entity}}
}

// PR7252
namespace test1 {
  namespace A { template<typename T> struct Base { typedef T t; }; } // expected-note 3{{member}}
  namespace B { template<typename T> struct Base { typedef T t; }; } // expected-note {{member found}}

  template<typename T> struct Derived : A::Base<char>, B::Base<int> {
    typename Derived::Base<float>::t x; // expected-error {{found in multiple base classes of different types}}
  };

  class X : A::Base<int> {}; // expected-note 2{{private}}
  class Y : A::Base<float> {};
  struct Z : A::Base<double> {};
  struct Use1 : X, Y {
    Base<double> b1; // expected-error {{private}}
    Use1::Base<double> b2; // expected-error {{private}}
  };
  struct Use2 : Z, Y {
    Base<double> b1;
    Use2::Base<double> b2;
  };
  struct Use3 : X, Z {
    Base<double> b1;
    Use3::Base<double> b2;
  };
}

namespace test2 {
  struct A { static int x; }; // expected-note 4{{member}}
  struct B { template<typename T> static T x(); }; // expected-note 4{{member}}
  struct C { template<typename T> struct x {}; }; // expected-note 3{{member}}
  struct D { template<typename T> static T x(); }; // expected-note {{member}}

  template<typename ...T> struct X : T... {};

  void f() {
    X<A, B>::x<int>(); // expected-error {{found in multiple base classes of different types}}
    X<A, C>::x<int>(); // expected-error {{found in multiple base classes of different types}}
    X<B, C>::x<int>(); // expected-error {{found in multiple base classes of different types}}
    X<A, B, C>::x<int>(); // expected-error {{found in multiple base classes of different types}}
    X<A, B, D>::x<int>(); // expected-error {{found in multiple base classes of different types}}
  }
}

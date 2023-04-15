// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2b %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename T> struct initializer_list {
    const T *p; size_t n;
    initializer_list(const T *p, size_t n);
  };
}

namespace dr1004 { // dr1004: 5
  template<typename> struct A {};
  template<typename> struct B1 {};
  template<template<typename> class> struct B2 {};
  template<typename X> void f(); // expected-note {{[with X = dr1004::A<int>]}}
  template<template<typename> class X> void f(); // expected-note {{[with X = dr1004::A]}}
  template<template<typename> class X> void g(); // expected-note {{[with X = dr1004::A]}}
  template<typename X> void g(); // expected-note {{[with X = dr1004::A<int>]}}
  struct C : A<int> {
    B1<A> b1a;
    B2<A> b2a;
    void h() {
      f<A>(); // expected-error {{ambiguous}}
      g<A>(); // expected-error {{ambiguous}}
    }
  };

  // This example (from the standard) is actually ill-formed, because
  // name lookup of "T::template A" names the constructor.
  template<class T, template<class> class U = T::template A> struct Third { }; // expected-error {{is a constructor name}}
  Third<A<int> > t; // expected-note {{in instantiation of default argument}}
}

namespace dr1042 { // dr1042: 3.5
#if __cplusplus >= 201402L
  // C++14 added an attribute that we can test the semantics of.
  using foo [[deprecated]] = int; // expected-note {{'foo' has been explicitly marked deprecated here}}
  foo f = 12; // expected-warning {{'foo' is deprecated}}
#elif __cplusplus >= 201103L
  // C++11 did not have any attributes that could be applied to an alias
  // declaration, so the best we can test is that we accept an empty attribute
  // list in this mode.
  using foo [[]] = int;
#endif
}

namespace dr1048 { // dr1048: 3.6
  struct A {};
  const A f();
  A g();
  typedef const A CA;
#if __cplusplus >= 201103L
  // ok: we deduce non-const A in each case.
  A &&a = [] (int n) {
    while (1) switch (n) {
      case 0: return f();
      case 1: return g();
      case 2: return A();
      case 3: return CA();
    }
  } (0);
#endif
}

namespace dr1054 { // dr1054: no
  // FIXME: Test is incomplete.
  struct A {} volatile a;
  void f() {
    // FIXME: This is wrong: an lvalue-to-rvalue conversion is applied here,
    // which copy-initializes a temporary from 'a'. Therefore this is
    // ill-formed because A does not have a volatile copy constructor.
    // (We might want to track this aspect under dr1383 instead?)
    a; // expected-warning {{assign into a variable to force a volatile load}}
  }
}

namespace dr1070 { // dr1070: 3.5
#if __cplusplus >= 201103L
  struct A {
    A(std::initializer_list<int>);
  };
  struct B {
    int i;
    A a;
  };
  B b = {1};
  struct C {
    std::initializer_list<int> a;
    B b;
    std::initializer_list<double> c;
  };
  C c = {};
#endif
}

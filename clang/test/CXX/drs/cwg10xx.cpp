// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename T> struct initializer_list {
    const T *p; size_t n;
    initializer_list(const T *p, size_t n);
  };
} // namespace std

namespace cwg1004 { // cwg1004: 5
  template<typename> struct A {};
  template<typename> struct B1 {};
  template<template<typename> class> struct B2 {};
  template<typename X> void f(); // #cwg1004-f-1
  template<template<typename> class X> void f(); // #cwg1004-f-2
  template<template<typename> class X> void g(); // #cwg1004-g-1
  template<typename X> void g(); // #cwg1004-g-2
  struct C : A<int> {
    B1<A> b1a;
    B2<A> b2a;
    void h() {
      f<A>();
      // expected-error@-1 {{call to 'f' is ambiguous}}
      //   expected-note@#cwg1004-f-1 {{candidate function [with X = cwg1004::A<int>]}}
      //   expected-note@#cwg1004-f-2 {{candidate function [with X = cwg1004::A]}}
      g<A>();
      // expected-error@-1 {{call to 'g' is ambiguous}}
      //   expected-note@#cwg1004-g-1 {{candidate function [with X = cwg1004::A]}}
      //   expected-note@#cwg1004-g-2 {{candidate function [with X = cwg1004::A<int>]}}
    }
  };

  // This example (from the standard) is actually ill-formed, because
  // name lookup of "T::template A" names the constructor.
  template<class T, template<class> class U = T::template A> struct Third { };
  // expected-error@-1 {{is a constructor name}}
  //   expected-note@#cwg1004-t {{in instantiation of default argument}}
  Third<A<int> > t; // #cwg1004-t
} // namespace cwg1004

namespace cwg1042 { // cwg1042: 3.5
#if __cplusplus >= 201402L
  // C++14 added an attribute that we can test the semantics of.
  using foo [[deprecated]] = int; // #cwg1042-using
  foo f = 12;
  // since-cxx14-warning@-1 {{'foo' is deprecated}}
  //   since-cxx14-note@#cwg1042-using {{'foo' has been explicitly marked deprecated here}}
#elif __cplusplus >= 201103L
  // C++11 did not have any attributes that could be applied to an alias
  // declaration, so the best we can test is that we accept an empty attribute
  // list in this mode.
  using foo [[]] = int;
#endif
} // namespace cwg1042

namespace cwg1048 { // cwg1048: 3.6
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
} // namespace cwg1048

namespace cwg1054 { // cwg1054: no
  // FIXME: Test is incomplete.
  struct A {} volatile a;
  void f() {
    // FIXME: This is wrong: an lvalue-to-rvalue conversion is applied here,
    // which copy-initializes a temporary from 'a'. Therefore this is
    // ill-formed because A does not have a volatile copy constructor.
    // (We might want to track this aspect under cwg1383 instead?)
    a;
    // expected-warning@-1 {{expression result unused; assign into a variable to force a volatile load}}
  }
} // namespace cwg1054

namespace cwg1070 { // cwg1070: 3.5
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
} // namespace cwg1070

// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++14-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++14-compat-pedantic -verify %s

#if __cplusplus < 201402L

// expected-no-diagnostics
// FIXME: C++11 features removed or changed in C++14?

#else

static_assert(true); // expected-warning {{incompatible with C++ standards before C++17}}

template<int ...N> int f() { return (N + ...); } // expected-warning {{incompatible with C++ standards before C++17}}

namespace [[]] NS_with_attr {} // expected-warning {{incompatible with C++ standards before C++17}}
enum { e [[]] }; // expected-warning {{incompatible with C++ standards before C++17}}

template<typename T = int> struct X {};
X x; // expected-warning {{class template argument deduction is incompatible with C++ standards before C++17; for compatibility, use explicit type name 'X<int>'}}

template<template<typename> class> struct Y {};
Y<X> yx; // ok, not class template argument deduction

template<typename T> void f(T t) {
  X x = t; // expected-warning {{incompatible}}
}

template<typename T> void g(T t) {
  typename T::X x = t; // expected-warning {{incompatible}}
}
struct A { template<typename T> struct X { X(T); }; };
void h(A a) { g(a); } // expected-note {{in instantiation of}}

#endif

// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T> using A = int; // expected-note 2{{previous}}
template<typename T> using A = char; // expected-error {{type alias template redefinition with different types ('char' vs 'int')}}
template<typename T1, typename T2> using A = T1; // expected-error {{too many template parameters in template redeclaration}}

template<typename T1, typename T2> using B = T1; // expected-note {{previous}}
template<typename T2, typename T1> using B = T1; // expected-error {{type alias template redefinition with different types}}


template<typename> struct S;
template<template<typename> class F> using FInt = F<int>;
template<typename X> using SXRInt = FInt<S<X>::template R>;
template<typename X> using SXRInt = typename S<X>::template R<int>; // ok, redeclaration.

template<template<typename> class> struct TT;

namespace FilterLookup {
  TT<A> f(); // expected-note {{previous declaration is here}}

  template<typename> using A = int;
  TT<A> f(); // expected-error {{functions that differ only in their return type cannot be overloaded}}
}

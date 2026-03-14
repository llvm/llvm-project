// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> void f0(T) { } // expected-note{{previous}}
template<class U> void f0(U) { } // expected-error{{redefinition}}

template<int I> void f0() { } // expected-note{{previous}}
template<int> void f0() { } // expected-error{{redefinition}}

typedef int INT;

template<template<class T, T Value1, INT> class X> 
  void f0() { } // expected-note{{previous}}
template<template<typename T, T Value1, int> class> 
  void f0() { } // expected-error{{redefinition}}

template<typename T>
struct MetaFun;

template<typename T>
  typename MetaFun<T*>::type f0(const T&) { while (1) {} } // expected-note{{previous}}
template<class U>
  typename MetaFun<U*>::type f0(const U&) { while (1) {} } // expected-error{{redefinition}}

// FIXME: We need canonicalization of expressions for this to work
// template<int> struct A { };
// template<int I> void f0(A<I>) { } // Xpected-note{{previous}}
// template<int J> void f0(A<J>) { } // Xpected-error{{redefinition}}

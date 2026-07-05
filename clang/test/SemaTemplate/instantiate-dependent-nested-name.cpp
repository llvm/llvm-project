// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
// PR4382
template<typename T> struct X { static const T A = 1; };
template<typename T, bool = X<T>::A> struct Y { typedef T A; };
template<typename T> struct Z { typedef typename Y<T>::A A; };
extern int x;
extern Z<int>::A x;

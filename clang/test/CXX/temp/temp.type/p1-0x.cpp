// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace Old {
  template<template<class> class TT> struct X { };
  template<class> struct Y { };
  template<class T> using Z = Y<T>;
  X<Y> y;
  X<Z> z;

  using SameType = decltype(y); // expected-note {{here}}
  using SameType = decltype(z); // expected-error {{different types}}
}

namespace New {
  template<class T> struct X { };
  template<class> struct Y { };
  template<class T> using Z = Y<T>;
  X<Y<int>> y;
  X<Z<int>> z;

  using SameType = decltype(y);
  using SameType = decltype(z); // ok
}

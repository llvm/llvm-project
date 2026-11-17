// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify=old -std=c++11 -triple x86_64-linux-gnu -fclang-abi-compat=23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=old -std=c++11 -triple x86_64-windows-msvc %s
// expected-no-diagnostics

namespace Old {
  template<template<class> class TT> struct X { };
  template<class> struct Y { };
  template<class T> using Z = Y<T>;
  X<Y> y;
  X<Z> z;

  using SameType = decltype(y); // old-note {{here}}
  using SameType = decltype(z); // old-error {{different types}}
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

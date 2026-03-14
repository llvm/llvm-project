// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

struct Q { enum F { f }; };

template<typename T> struct A : Q {
  enum E { e } E = e;

  using Q::F;
  Q::F F = f;
};
A<int> a = {};

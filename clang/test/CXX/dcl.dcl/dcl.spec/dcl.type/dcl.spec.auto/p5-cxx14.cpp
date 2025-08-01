// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14

struct A {
  template<int N>
  static constexpr auto x = N;

  template<>
  constexpr auto x<1> = 1;

  template<>
  static constexpr auto x<2> = 2; // expected-warning{{explicit specialization cannot have a storage class}}
};

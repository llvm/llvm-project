// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++23 %s -verify

// expected-no-diagnostics

struct A {};
using CA = const A;

struct S1 : CA {
  constexpr S1() : CA() {}
};

struct S2 : A {
  constexpr S2() : CA() {}
};

struct S3 : CA {
  constexpr S3() : A() {}
};

struct Int {};

template <class _Hp>
struct __tuple_leaf : _Hp {
  constexpr __tuple_leaf() : _Hp() {}
};

constexpr __tuple_leaf<const Int> t;

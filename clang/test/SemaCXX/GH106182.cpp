// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class Fn> struct A {
  constexpr A(Fn) {};
};

template <template <class> class S>
 void create_unique()
   requires (S{0}, true);

template <template <class> class S>
 void create_unique()
   requires (S{0}, true) {}

template void create_unique<A>();

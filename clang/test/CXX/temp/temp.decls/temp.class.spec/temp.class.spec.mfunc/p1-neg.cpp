// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, int N>
struct A;

template<typename T>
struct A<T*, 2> {
  void f0();
  void f1();
  void f2();
};

template<>
struct A<int, 1> {
  void g0();
};

// FIXME: We should produce diagnostics pointing out the
// non-matching candidates.
template<typename T, int N>
void A<T*, 2>::f0() { } // expected-error{{does not refer into a class, class template or class template partial specialization}}

template<typename T, int N>
void A<T, N>::f1() { } // expected-error{{out-of-line definition}}

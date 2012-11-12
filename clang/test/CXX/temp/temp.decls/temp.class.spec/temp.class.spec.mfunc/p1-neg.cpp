// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, int N>
struct A;

template<typename T> // expected-note{{previous template declaration}}
struct A<T*, 2> {
  void f0();
  void f1();
  void f2();
};

template<>
struct A<int, 1> {
  void g0();
};

// FIXME: We should probably give more precise diagnostics here, but the
// diagnostics we give aren't terrible.
// FIXME: why not point to the first parameter that's "too many"?
template<typename T, int N> // expected-error{{too many template parameters}}
void A<T*, 2>::f0() { }

template<typename T, int N>
void A<T, N>::f1() { } // expected-error{{out-of-line definition}}

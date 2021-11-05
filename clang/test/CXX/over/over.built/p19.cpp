// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(int i, float f, int* pi, T* pt, T t) {
  i = i;
  i *= i;
  i /= i;
  i += i;
  i -= i;
  i -= f;
  i -= pi; // expected-error {{invalid operands}}
  i -= pt; // FIXME
  i -= t;

  f = f;
  f *= f;
  f /= f;
  f += f;
  f -= f;
  f -= i;
  f -= pi; // expected-error {{invalid operands}}
  f -= pt; // FIXME
  f -= t;
}

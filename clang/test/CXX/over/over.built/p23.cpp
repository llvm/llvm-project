// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T, typename U>
void f(int i, float f, bool b, int* pi, T* pt, T t) {
  i %= 3;
  f %= 3;  // expected-error {{invalid operands}}
  b %= 3;
  pi %= 3; // expected-error {{invalid operands}}
  pt %= 3; // FIXME
  t %= 3;

  i &= 3;
  f &= 3;  // expected-error {{invalid operands}}
  b &= 3;
  pi &= 3; // expected-error {{invalid operands}}
  pt &= 3; // FIXME
  t &= 3;

  i ^= 3;
  f ^= 3;  // expected-error {{invalid operands}}
  b ^= 3;
  pi ^= 3; // expected-error {{invalid operands}}
  pt ^= 3; // FIXME
  t ^= 3;

  i |= 3;
  f |= 3;  // expected-error {{invalid operands}}
  b |= 3;
  pi |= 3; // expected-error {{invalid operands}}
  pt |= 3; // FIXME
  t |= 3;

  i <<= 3;
  f <<= 3;  // expected-error {{invalid operands}}
  b <<= 3;
  pi <<= 3; // expected-error {{invalid operands}}
  pt <<= 3; // FIXME
  t <<= 3;

  i >>= 3;
  f >>= 3;  // expected-error {{invalid operands}}
  b >>= 3;
  pi >>= 3; // expected-error {{invalid operands}}
  pt >>= 3; // FIXME
  t >>= 3;
}

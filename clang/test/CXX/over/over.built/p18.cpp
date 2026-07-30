// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T, typename U>
void f(int i, float f, bool b, int* pi, T* pt, T t) {
  (void)(i % 3);
  (void)(f % 3);  // expected-error {{invalid operands}}
  (void)(b % 3);
  (void)(pi % 3); // expected-error {{invalid operands}}
  (void)(pt % 3); // FIXME
  (void)(t % 3);
  (void)(3 % i);
  (void)(3 % f);  // expected-error {{invalid operands}}
  (void)(3 % b);
  (void)(3 % pi); // expected-error {{invalid operands}}
  (void)(3 % pt); // FIXME
  (void)(3 % t);

  (void)(i & 3);
  (void)(f & 3);  // expected-error {{invalid operands}}
  (void)(b & 3);
  (void)(pi & 3); // expected-error {{invalid operands}}
  (void)(pt & 3); // FIXME
  (void)(t & 3);
  (void)(3 & i);
  (void)(3 & f);  // expected-error {{invalid operands}}
  (void)(3 & b);
  (void)(3 & pi); // expected-error {{invalid operands}}
  (void)(3 & pt); // FIXME
  (void)(3 & t);

  (void)(i ^ 3);
  (void)(f ^ 3);  // expected-error {{invalid operands}}
  (void)(b ^ 3);
  (void)(pi ^ 3); // expected-error {{invalid operands}}
  (void)(pt ^ 3); // FIXME
  (void)(t ^ 3);
  (void)(3 ^ i);
  (void)(3 ^ f);  // expected-error {{invalid operands}}
  (void)(3 ^ b);
  (void)(3 ^ pi); // expected-error {{invalid operands}}
  (void)(3 ^ pt); // FIXME
  (void)(3 ^ t);

  (void)(i | 3);
  (void)(f | 3);  // expected-error {{invalid operands}}
  (void)(b | 3);
  (void)(pi | 3); // expected-error {{invalid operands}}
  (void)(pt | 3); // FIXME
  (void)(t | 3);
  (void)(3 | i);
  (void)(3 | f);  // expected-error {{invalid operands}}
  (void)(3 | b);
  (void)(3 | pi); // expected-error {{invalid operands}}
  (void)(3 | pt); // FIXME
  (void)(3 | t);

  (void)(i << 3);
  (void)(f << 3);  // expected-error {{invalid operands}}
  (void)(b << 3);
  (void)(pi << 3); // expected-error {{invalid operands}}
  (void)(pt << 3); // FIXME
  (void)(t << 3);
  (void)(3 << i);
  (void)(3 << f);  // expected-error {{invalid operands}}
  (void)(3 << b);
  (void)(3 << pi); // expected-error {{invalid operands}}
  (void)(3 << pt); // FIXME
  (void)(3 << t);

  (void)(i >> 3);
  (void)(f >> 3);  // expected-error {{invalid operands}}
  (void)(b >> 3);
  (void)(pi >> 3); // expected-error {{invalid operands}}
  (void)(pt >> 3); // FIXME
  (void)(t >> 3);
  (void)(3 >> i);
  (void)(3 >> f);  // expected-error {{invalid operands}}
  (void)(3 >> b);
  (void)(3 >> pi); // expected-error {{invalid operands}}
  (void)(3 >> pt); // FIXME
  (void)(3 >> t);
}

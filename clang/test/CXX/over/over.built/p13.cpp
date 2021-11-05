// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(int i, float f, bool b, char c, int* pi, T* pt) {
  (void)(i*i);
  (void)(i*f);
  (void)(i*b);
  (void)(i*c);
  (void)(i*pi); // expected-error {{invalid operands to binary expression}}
  (void)(i*pt); // FIXME

  (void)(i/i);
  (void)(i/f);
  (void)(i/b);
  (void)(i/c);
  (void)(i/pi); // expected-error {{invalid operands to binary expression}}
  (void)(i/pt); // FIXME

  (void)(i-i);
  (void)(i-f);
  (void)(i-b);
  (void)(i-c);
  (void)(i-pi); // expected-error {{invalid operands to binary expression}}
  (void)(i-pt); // FIXME

  (void)(i<i);
  (void)(i<f);
  (void)(i<b);
  (void)(i<c);
  (void)(i<pi); // expected-error {{comparison between pointer and integer}}
  (void)(i<pt); // FIXME

  (void)(i==i);
  (void)(i==f);
  (void)(i==b);
  (void)(i==c);
  (void)(i==pi); // expected-error {{comparison between pointer and integer}}
  (void)(i==pt); // FIXME
}


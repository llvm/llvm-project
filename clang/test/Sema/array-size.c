// RUN: %clang_cc1 -triple i686-apple-darwin -verify %s

void f() {
  int x0[1073741824]; // expected-error{{array is too large}}
  int x1[1073741824 + 1]; // expected-error{{array is too large}}
  int x2[(unsigned)1073741824]; // expected-error{{array is too large}}
  int x3[(unsigned)1073741824 + 1]; // expected-error{{array is too large}}
  int x4[1073741824 - 1];
}


// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -Wno-sometimes-uninitialized -verify %s

void foo(const int &);

int f(bool a) {
  int v;
  if (a) {
    foo(v); // expected-warning {{variable 'v' is uninitialized when passed as a const reference argument here}}
    v = 5;
  }
  return v;
}

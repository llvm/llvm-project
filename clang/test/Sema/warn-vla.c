// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify -Wvla %s
// RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -Wvla %s

void test1(int n) {
  int v[n]; // expected-warning {{variable length array}}
}

void test2(int n, int v[n]) { // expected-warning {{variable length array}}
}

void test3(int n, int v[n]); // expected-warning {{variable length array}}


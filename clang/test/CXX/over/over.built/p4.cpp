// RUN: %clang_cc1 -std=c++17 -verify %s -Wno-tautological-compare

void f(int i, bool b) {
  (void)++i;
  (void)i++;

  (void)++b; // expected-error {{ISO C++17 does not allow incrementing expression of type bool}}
  (void)b++; // expected-error {{ISO C++17 does not allow incrementing expression of type bool}}
}


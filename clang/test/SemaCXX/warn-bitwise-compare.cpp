// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-bitwise-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -Wno-unused %s

void test(int x) {
  bool b1 = (8 & x) == 3;
  // expected-warning@-1 {{bitwise comparison always evaluates to false}}
  bool b2 = x | 5;
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
  bool b3 = (x | 5);
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
  bool b4 = !!(x | 5);
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
}

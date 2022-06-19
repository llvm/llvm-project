// RUN: %clang_cc1 %s -triple avr -target-cpu at90s8515 -fsyntax-only -verify

int foo(int n) {
  static __flash  const int a0[] = {4, 6}; // OK
  static __flash1 const int a1[] = {4, 6}; // expected-error {{unknown type name '__flash1'}}
  static __flash2 const int a2[] = {4, 6}; // expected-error {{unknown type name '__flash2'}}
  static __flash3 const int a3[] = {4, 6}; // expected-error {{unknown type name '__flash3'}}
  static __flash4 const int a4[] = {4, 6}; // expected-error {{unknown type name '__flash4'}}
  static __flash5 const int a5[] = {4, 6}; // expected-error {{unknown type name '__flash5'}}
  // TODO: It would be better to report "'__flash5' is not supported on at908515".
  return a0[n] + a1[n];
}

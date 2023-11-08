// RUN: %clang_cc1 -verify %s

struct A {
  enum E : unsigned {};
  enum E2 : int {};
  [[clang::preferred_type(E)]] unsigned b : 2;
  [[clang::preferred_type(E)]] int b2 : 2;
  [[clang::preferred_type(E2)]] const unsigned b3 : 2;
  [[clang::preferred_type(bool)]] unsigned b4 : 1;
  [[clang::preferred_type(bool)]] unsigned b5 : 2;
  [[clang::preferred_type()]] unsigned b6 : 2;
  // expected-error@-1 {{'preferred_type' attribute takes one argument}}
  [[clang::preferred_type]] unsigned b7 : 2;
  // expected-error@-1 {{'preferred_type' attribute takes one argument}}
  [[clang::preferred_type(E, int)]] unsigned b8 : 2;
  // expected-error@-1 {{expected ')'}}
  // expected-error@-2 {{expected ','}}
  // expected-warning@-3 {{unknown attribute 'int' ignored}}
};

// RUN: %clang_cc1 -verify %s

struct A {
  enum E : unsigned {};
  [[clang::preferred_type(E)]] unsigned b : 2;
  [[clang::preferred_type(E)]] int b2 : 2;
  // expected-warning@-1 {{underlying type 'unsigned int' of enumeration 'E' doesn't match bit-field type 'int'}}
  [[clang::preferred_type(E)]] const unsigned b3 : 2;
  [[clang::preferred_type(bool)]] unsigned b4 : 1;
  [[clang::preferred_type(bool)]] unsigned b5 : 2;
  // expected-warning@-1 {{bit-field that holds a boolean value should have width of 1 instead of 2}}
  [[clang::preferred_type()]] unsigned b6 : 2;
  // expected-error@-1 {{'preferred_type' attribute takes one argument}}
  [[clang::preferred_type]] unsigned b7 : 2;
  // expected-error@-1 {{'preferred_type' attribute takes one argument}}
  [[clang::preferred_type(E, int)]] unsigned b8 : 2;
  // expected-error@-1 {{expected ')'}}
  // expected-error@-2 {{expected ','}}
  // expected-warning@-3 {{unknown attribute 'int' ignored}}
};
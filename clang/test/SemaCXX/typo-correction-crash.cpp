// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
auto check1() {
  return 1;
  return s; // expected-error {{use of undeclared identifier 's'}}
}

int test = 11; // expected-note {{'test' declared here}}
auto check2() {
  return "s";
  return tes; // expected-error {{use of undeclared identifier 'tes'; did you mean 'test'?}}
}

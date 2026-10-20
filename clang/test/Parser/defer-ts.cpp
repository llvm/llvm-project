// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fdefer-ts -verify %s

void f() {
  _Defer {} // expected-error {{use of undeclared identifier '_Defer'}}
}

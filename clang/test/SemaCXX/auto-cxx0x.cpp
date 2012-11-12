// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}

// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wc++11-compat 
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
  int auto b; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
  auto c; // expected-warning {{C++11 extension}} expected-error {{requires an initializer}}
  static auto d = 0; // expected-warning {{C++11 extension}}
  auto static e = 0; // expected-warning {{C++11 extension}}
}

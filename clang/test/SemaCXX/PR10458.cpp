// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98

void f() {
  int arr[] = { 1, 2, 3 };
  for (auto &i : arr) { // expected-warning {{'auto' type specifier is a C++11 extension}} expected-warning {{range-based for loop is a C++11 extension}}
  }
}

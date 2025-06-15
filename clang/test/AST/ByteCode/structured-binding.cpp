// RUN: %clang_cc1 -std=c++17 -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++17 -verify %s

void f1() {
  int arr[2] = {};
  auto [a, b] = arr;
  static_assert(&a != &b);  // expected-no-diagnostics
}

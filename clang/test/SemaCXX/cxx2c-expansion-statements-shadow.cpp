// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -Wshadow -verify
// expected-no-diagnostics

// Test that -Wshadow doesn't fire on implicit variable declarations
// introduced in the expansion of an expansion statement.

void f() {
  int a[4];
  template for (int __u0; auto x : a) {}
  template for (auto x : a) { int __u0; }
}

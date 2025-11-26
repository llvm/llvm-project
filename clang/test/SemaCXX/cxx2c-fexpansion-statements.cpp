// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fexpansion-limit=0 -verify
// expected-no-diagnostics

// Test that passing =0 disables the limit.

void big() {
  int ok[500];
  template for (auto x : ok) {}
}

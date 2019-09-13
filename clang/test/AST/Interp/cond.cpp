// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fforce-experimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++17 -fsyntax-only %s -verify
// expected-no-diagnostics

constexpr int cond_then_else(int a, int b) {
  if (a < b) {
    return b - a;
  } else {
    return a - b;
  }
}

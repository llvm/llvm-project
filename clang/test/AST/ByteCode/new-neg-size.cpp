// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify %s

constexpr void f() {
  int a = -1;
  int *b = new int[a]; // expected-note {{cannot allocate array with negative size in a constant expression}}
}
// expected-error@-4 {{constexpr function never produces a constant expression}}

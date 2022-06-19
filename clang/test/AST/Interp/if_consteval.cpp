// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++2b -fsyntax-only %s -verify
// expected-no-diagnostics

constexpr void f() {
  int i = 0;
  if consteval {
    i = 1;
  }
  else {
    i = 2;
  }

  if consteval {
    i = 1;
  }

  if !consteval {
    i = 1;
  }

  if !consteval {
    i = 1;
  }
  else {
    i = 1;
  }
}

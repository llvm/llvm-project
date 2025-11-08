// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 -x c %s

// Test that 'auto' cannot be combined with a type specifier in C++.
void f() {
  auto int x = 1; // expected-error {{'auto' cannot be combined with a type specifier in C++}}
}

// c23-no-diagnostics

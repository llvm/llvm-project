// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

// Test that 'auto' cannot be combined with a type specifier in C++.
void f() {
  auto int x = 1;        // expected-error {{'auto' cannot be combined with a type specifier in C++}}
  auto char c = 'a';    // expected-error {{'auto' cannot be combined with a type specifier in C++}}
  auto float f = 1.0f;  // expected-error {{'auto' cannot be combined with a type specifier in C++}}
  auto double d = 1.0;   // expected-error {{'auto' cannot be combined with a type specifier in C++}}
  auto long l = 1L;     // expected-error {{'auto' cannot be combined with a type specifier in C++}}
}

// Test that regular 'auto' (type deduction) still works in C++.
void h() {
  auto x = 1;
  auto y = 2.0;
  auto z = 'c';
}


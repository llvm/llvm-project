// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

[[clang::reinitializes]] int a; // expected-error {{'reinitializes' attribute only applies to non-static non-const member functions}}

[[clang::reinitializes]] void f(); // expected-error {{only applies to}}

struct A {
  [[clang::reinitializes]] void foo();
  __attribute__((reinitializes)) void gnu_foo();
  [[clang::reinitializes]] void bar() const; // expected-error {{only applies to}}
  [[clang::reinitializes]] static void baz(); // expected-error {{only applies to}}
  [[clang::reinitializes]] int a; // expected-error {{only applies to}}

  [[clang::reinitializes("arg")]] void qux(); // expected-error {{'reinitializes' attribute takes no arguments}}
};

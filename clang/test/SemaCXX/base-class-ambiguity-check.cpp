// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template <typename T> class Foo {
  struct Base : T {};

  // Test that this code no longer causes a crash in Sema. rdar://23291875
  struct Derived : Base, T {};
};

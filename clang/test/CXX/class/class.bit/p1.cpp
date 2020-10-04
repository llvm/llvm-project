// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// Test various bit-field member declarations.
constexpr int foo() { return 1; }
struct A {
  int a [[]] : 1;
  int b, [[]] : 0; // expected-error {{an attribute list cannot appear here}}
  int [[]] : 0; // OK, attribute applies to the type.
  int [[]] c : 1; // OK, attribute applies to the type.
  int : 2 = 1; // expected-error {{anonymous bit-field cannot have a default member initializer}}
  int : 0 { 1 }; // expected-error {{anonymous bit-field cannot have a default member initializer}}
  int : 0, d : 1 = 1;
  int : 1 = 12, e : 1; // expected-error {{anonymous bit-field cannot have a default member initializer}}
  int : 0, f : 1 = 1;
  int g [[]] : 1 = 1;
  int h [[]] : 1 {1};
  int i : foo() = foo();
  int j, [[]] k; // expected-error {{an attribute list cannot appear here}}
};

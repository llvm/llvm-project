// RUN: %clang_cc1 -fsyntax-only -verify -xobjective-c++-header %s

template <typename, typename>
class C {};

class B {
  p // expected-error {{unknown type name 'p'}}
 private: // expected-error {{'private' is a keyword in Objective-C++}}
  void f() {} // expected-error {{expected '(' for function-style cast or type construction}}
  C<int, decltype(f)> c; // expected-error {{use of undeclared identifier 'f'}}
  // expected-error@-1 {{expected member name}}
};

// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
template <typename A, decltype(new A)>
struct b;
struct d {
  static auto c = ;              // expected-error{{expected expression}}
                                 // expected-error@-1 {{declaration of variable 'c' with deduced type 'auto' requires an initializer}}

  decltype(b<decltype(c), int>); // expected-error{{expected '(' for function-style cast or type construction}}
};

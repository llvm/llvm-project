// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
template <typename A, decltype(new A)> // expected-error{{new expression for type 'auto' requires a constructor argument}}
struct b;
struct d {
  static auto c = ;              // expected-error{{expected expression}}
  decltype(b<decltype(c), int>); // expected-error{{expected '(' for function-style cast or type construction}}
                                 // expected-note@-1{{while substituting prior template arguments into non-type template parameter [with A = auto]}}
};

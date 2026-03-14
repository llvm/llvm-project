// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// A constructor shall not be declared with a ref-qualifier.
struct X {
  X() &; // expected-error{{ref-qualifier '&' is not allowed on a constructor}}
  X(int) &&; // expected-error{{ref-qualifier '&&' is not allowed on a constructor}}
};

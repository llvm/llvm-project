// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// A destructor shall not be declared with a ref-qualifier.
struct X {
  ~X() &; // expected-error{{ref-qualifier '&' is not allowed on a destructor}}
};

struct Y {
  ~Y() &&; // expected-error{{ref-qualifier '&&' is not allowed on a destructor}}
};

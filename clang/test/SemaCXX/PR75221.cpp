// RUN: %clang_cc1 -verify -std=c++11 -fsyntax-only %s

template <class T> using foo = struct foo { // expected-error {{'foo' cannot be defined in a type alias template}}
  T size = 0;
};
foo a;

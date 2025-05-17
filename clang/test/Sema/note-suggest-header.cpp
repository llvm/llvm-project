// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(void) {
  int_val2 = 0; // expected-error{{use of undeclared identifier}}
  sin(0); // expected-error{{use of undeclared identifier 'sin'}} \
          // expected-note{{perhaps `#include <cmath>` is needed?}}

  std::cout << "Hello world\n"; // expected-error{{use of undeclared identifier 'std'}} \
                                // expected-note{{perhaps `#include <iostream>` is needed?}}
}
// RUN: %clang_cc1 -std=c++20 -verify %s

namespace P1972 {
void f(int) requires false; // expected-note 4{{because 'false' evaluated to false}} \
                            // expected-note{{constraints not satisfied}}
void g() {
  f(0);                      // expected-error{{no matching function for call to 'f'}}
  void (*p1)(int) = f;       // expected-error{{invalid reference to function 'f': constraints not satisfied}}
  void (*p21)(int) = &f;     // expected-error{{invalid reference to function 'f': constraints not satisfied}}
  decltype(f) *p2 = nullptr; // expected-error{{invalid reference to function 'f': constraints not satisfied}}
}
}

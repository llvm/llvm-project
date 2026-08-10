// RUN: %clang_cc1 -std=c++20 -verify %s

namespace P1972 {
template <typename T>
struct S {
  static void f(int)
    requires false; // expected-note 4{{because 'false' evaluated to false}}
};
void g() {
  S<int>::f(0);                      // expected-error{{invalid reference to function 'f': constraints not satisfied}}
  void (*p1)(int) = S<int>::f;       // expected-error{{invalid reference to function 'f': constraints not satisfied}}
  void (*p21)(int) = &S<int>::f;     // expected-error{{invalid reference to function 'f': constraints not satisfied}}
  decltype(S<int>::f) *p2 = nullptr; // expected-error{{invalid reference to function 'f': constraints not satisfied}}
}
}

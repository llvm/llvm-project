// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

export module m3;
export template <class> struct X {
  template <class Self> friend void f(Self &&self) {
    (Self&)self; // expected-warning {{expression result unused}}
  }
};
void g() { f(X<void>{}); } // expected-note {{in instantiation of function template specialization}}

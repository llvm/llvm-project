// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(int i, float f, bool b, char c, int* pi, T* pt) {
  (void)~i;
  (void)~f; // expected-error {{invalid argument type}}
  (void)~b;
  (void)~c;
  (void)~pi; // expected-error {{invalid argument type}}
  (void)~pt; // expected-error {{invalid argument type}}
}


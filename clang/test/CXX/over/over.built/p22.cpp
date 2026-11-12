// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(int* pi, T* pt, T t) {
  pi += 3;
  pi += pi; // expected-error {{invalid operands}}
  pt += 3;
  pi += t;
  pi += pt; // FIXME
  pt += pi; //FIXME
  pt += pt; //FIXME
}

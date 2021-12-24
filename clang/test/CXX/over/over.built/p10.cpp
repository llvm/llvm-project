// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A{};

template <typename T>
void f(int i, float f, bool b, char c, int* pi, A* pa, T* pt) {
  (void)+i;
  (void)-i;
  (void)+f;
  (void)-f;
  (void)+b;
  (void)-b;
  (void)+c;
  (void)-c;

  (void)-pi; // expected-error {{invalid argument type}}
  (void)-pa; // expected-error {{invalid argument type}}
  (void)-pt; // FIXME: we should be able to give an error here.
}


// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -verify %s
// expected-no-diagnostics

template <typename T>
void f() {
  T().~T();
}

template <typename T>
void f1(T *f) {
  f->~T();
  (*f).~T();
}

using mat4 = float __attribute__((matrix_type(4, 4)));
using mat4i = int __attribute__((matrix_type(4, 4)));

template <typename T>
using mat4_t = T __attribute__((matrix_type(4, 4)));

void g() {
  f<mat4>();
  f<mat4i>();
  f<mat4_t<double>>();
}

void g2(mat4* m1, mat4i* m2, mat4_t<double>* m3) {
  f1(m1);
  f1(m2);
  f1(m3);
}

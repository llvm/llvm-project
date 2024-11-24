// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++11 -verify %s
// expected-no-diagnostics

template <typename T>
void f() {
  T().~T();
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

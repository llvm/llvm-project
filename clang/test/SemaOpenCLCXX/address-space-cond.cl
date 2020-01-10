// RUN: %clang_cc1 %s -cl-std=clc++ -pedantic -verify

namespace PointerRvalues {

void f(__global int *__constant *a, const __global int *__constant *b) {
  using T = decltype(true ? +a : +b);
  using T = const __global int *const __constant *;
}

void g(const __global int *a, __generic int *b) {
  using T = decltype(true ? +a : +b);
  using T = const __generic int *;
}

void h(const __global int **a, __generic int **b) {
  using T = decltype(true ? +a : +b); // expected-error {{incompatible operand types}}
}

void i(__global int **a, __generic int **b) {
  using T = decltype(true ? +a : +b); // expected-error {{incompatible operand types}}
}

}

// RUN: %clang_cc1 -x c++ -verify %s
// expected-no-diagnostics

template <int Unroll> void foo() {
  #pragma unroll Unroll
  for (int i = 0; i < Unroll; ++i);

  #pragma GCC unroll Unroll
  for (int i = 0; i < Unroll; ++i);
}

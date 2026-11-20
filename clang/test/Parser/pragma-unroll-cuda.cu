// RUN: %clang_cc1 -x cuda -verify -Werror %s
// RUN: %clang_cc1 -x hip -verify -Werror %s

// The argument to '#pragma unroll' may be parenthesized in CUDA/HIP. nvcc
// accepts both forms and generates identical code, and the CUDA C++ Programming
// Guide documents the parenthesized form, so this must not diagnose.

// expected-no-diagnostics

void test(int *List, int Length) {
#pragma unroll(4)
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }

#pragma unroll 4
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }

#pragma unroll (2 + 2)
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }
}

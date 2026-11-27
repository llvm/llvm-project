// RUN: %clang_cc1 -x cuda -verify -Werror -Wcuda-compat %s
// RUN: %clang_cc1 -x cuda -triple nvptx64-nvidia-cuda -fcuda-is-device -verify -Werror -Wcuda-compat %s
// RUN: %clang_cc1 -x hip -verify -Werror -Wcuda-compat %s
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -fcuda-is-device -verify -Werror -Wcuda-compat %s

// The argument to '#pragma unroll' may be parenthesized in CUDA/HIP. nvcc
// accepts both forms and generates identical code, and the CUDA C++ Programming
// Guide documents the parenthesized form (including `#pragma unroll (N + 1)` on
// a template parameter), so this must not diagnose.

// cc1 does not pull in the CUDA headers, so spell the attribute out.
#define __device__ __attribute__((device))

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

__device__ void device_test(int *List, int Length) {
#pragma unroll(4)
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }
}

template <int N>
void template_test(int *List, int Length) {
#pragma unroll (N + 1)
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }
}

template void template_test<3>(int *, int);

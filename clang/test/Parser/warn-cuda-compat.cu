// RUN: %clang_cc1 -Wno-cuda-compat -Werror %s
// RUN: %clang_cc1 -Wcuda-compat -verify %s
// RUN: %clang_cc1 -x c++ -Wcuda-compat -Werror %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length) {
/* expected-warning {{argument to '#pragma unroll' should not be in parentheses in CUDA C/C++}} */#pragma unroll(4)
  for (int i = 0; i < Length; ++i) {
    List[i] = i;
  }
}

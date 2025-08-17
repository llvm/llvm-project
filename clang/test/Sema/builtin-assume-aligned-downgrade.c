// RUN: %clang_cc1 -fsyntax-only -Wno-int-conversion -triple x86_64-linux -verify %s

// Check that the pointer->int conversion error is not downgradable for the
// pointer argument to __builtin_assume_aligned.

int test(int *a, int b) {
  a = (int *)__builtin_assume_aligned(b, 32); // expected-error {{non-pointer argument to '__builtin_assume_aligned' is not allowed}}
  int *y = __builtin_assume_aligned(1, 1); // expected-error {{non-pointer argument to '__builtin_assume_aligned' is not allowed}}
}

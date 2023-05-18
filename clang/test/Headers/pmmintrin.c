// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics

#if defined(i386) || defined(__x86_64__)
#include <pmmintrin.h>

int __attribute__((__target__(("sse3")))) foo(int a) {
  _mm_mwait(0, 0);
  return 4;
}
#endif

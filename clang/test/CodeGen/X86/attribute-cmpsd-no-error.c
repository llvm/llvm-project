// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown-emit-llvm -o /dev/null -verify
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown-emit-llvm -o /dev/null -verify

// expected-no-diagnostics

#include <immintrin.h>

__attribute__((target("avx")))
__m128 test(__m128 a, __m128 b) {
  return _mm_cmp_ps(a, b, 14);
}

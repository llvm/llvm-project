// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// No warnings.
extern __m256i a;
int __attribute__((target("avx"))) bar(__m256i a) {
  return _mm256_extract_epi32(a, 3);
}

int baz() {
  return bar(a);
}

int __attribute__((target("avx"))) qq_avx(__m256i a) {
  return _mm256_extract_epi32(a, 3);
}

int qq_noavx() {
  return 0;
}

extern __m256i a;
int qq() {
  if (__builtin_cpu_supports("avx"))
    return qq_avx(a);
  else
    return qq_noavx();
}

// Test that fma and fma4 are both separately and combined valid for an fma intrinsic.
__m128 __attribute__((target("fma"))) fma_1(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddps(a, b, c);
}

__m128 __attribute__((target("fma4"))) fma_2(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddps(a, b, c);
}

__m128 __attribute__((target("fma,fma4"))) fma_3(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddps(a, b, c);
}

void verifyfeaturestrings() {
  (void)__builtin_cpu_supports("cmov");
  (void)__builtin_cpu_supports("mmx");
  (void)__builtin_cpu_supports("popcnt");
  (void)__builtin_cpu_supports("sse");
  (void)__builtin_cpu_supports("sse2");
  (void)__builtin_cpu_supports("sse3");
  (void)__builtin_cpu_supports("ssse3");
  (void)__builtin_cpu_supports("sse4.1");
  (void)__builtin_cpu_supports("sse4.2");
  (void)__builtin_cpu_supports("avx");
  (void)__builtin_cpu_supports("avx2");
  (void)__builtin_cpu_supports("sse4a");
  (void)__builtin_cpu_supports("fma4");
  (void)__builtin_cpu_supports("xop");
  (void)__builtin_cpu_supports("fma");
  (void)__builtin_cpu_supports("avx512f");
  (void)__builtin_cpu_supports("bmi");
  (void)__builtin_cpu_supports("bmi2");
  (void)__builtin_cpu_supports("aes");
  (void)__builtin_cpu_supports("pclmul");
  (void)__builtin_cpu_supports("avx512vl");
  (void)__builtin_cpu_supports("avx512bw");
  (void)__builtin_cpu_supports("avx512dq");
  (void)__builtin_cpu_supports("avx512cd");
  (void)__builtin_cpu_supports("avx512er");
  (void)__builtin_cpu_supports("avx512pf");
  (void)__builtin_cpu_supports("avx512vbmi");
  (void)__builtin_cpu_supports("avx512ifma");
}

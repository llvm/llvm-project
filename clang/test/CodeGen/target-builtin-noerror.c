// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// No warnings.
extern __m256i a;
int __attribute__((target("avx"))) bar() {
  return _mm256_extract_epi32(a, 3);
}

int baz() {
  return bar();
}

int __attribute__((target("avx"))) qq_avx() {
  return _mm256_extract_epi32(a, 3);
}

int qq_noavx() {
  return 0;
}

extern __m256i a;
int qq() {
  if (__builtin_cpu_supports("avx"))
    return qq_avx();
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
  (void)__builtin_cpu_supports("avx5124vnniw");
  (void)__builtin_cpu_supports("avx5124fmaps");
  (void)__builtin_cpu_supports("avx512vpopcntdq");
  (void)__builtin_cpu_supports("avx512vbmi2");
  (void)__builtin_cpu_supports("gfni");
  (void)__builtin_cpu_supports("vpclmulqdq");
  (void)__builtin_cpu_supports("avx512vnni");
  (void)__builtin_cpu_supports("avx512bitalg");
  (void)__builtin_cpu_supports("avx512bf16");
  (void)__builtin_cpu_supports("avx512vp2intersect");
}

void verifycpustrings() {
  (void)__builtin_cpu_is("alderlake");
  (void)__builtin_cpu_is("amd");
  (void)__builtin_cpu_is("amdfam10h");
  (void)__builtin_cpu_is("amdfam15h");
  (void)__builtin_cpu_is("amdfam17h");
  (void)__builtin_cpu_is("atom");
  (void)__builtin_cpu_is("barcelona");
  (void)__builtin_cpu_is("bdver1");
  (void)__builtin_cpu_is("bdver2");
  (void)__builtin_cpu_is("bdver3");
  (void)__builtin_cpu_is("bdver4");
  (void)__builtin_cpu_is("bonnell");
  (void)__builtin_cpu_is("broadwell");
  (void)__builtin_cpu_is("btver1");
  (void)__builtin_cpu_is("btver2");
  (void)__builtin_cpu_is("cannonlake");
  (void)__builtin_cpu_is("cascadelake");
  (void)__builtin_cpu_is("cooperlake");
  (void)__builtin_cpu_is("core2");
  (void)__builtin_cpu_is("corei7");
  (void)__builtin_cpu_is("goldmont");
  (void)__builtin_cpu_is("goldmont-plus");
  (void)__builtin_cpu_is("haswell");
  (void)__builtin_cpu_is("icelake-client");
  (void)__builtin_cpu_is("icelake-server");
  (void)__builtin_cpu_is("intel");
  (void)__builtin_cpu_is("istanbul");
  (void)__builtin_cpu_is("ivybridge");
  (void)__builtin_cpu_is("knl");
  (void)__builtin_cpu_is("knm");
  (void)__builtin_cpu_is("nehalem");
  (void)__builtin_cpu_is("rocketlake");
  (void)__builtin_cpu_is("sandybridge");
  (void)__builtin_cpu_is("shanghai");
  (void)__builtin_cpu_is("silvermont");
  (void)__builtin_cpu_is("skylake");
  (void)__builtin_cpu_is("skylake-avx512");
  (void)__builtin_cpu_is("slm");
  (void)__builtin_cpu_is("tigerlake");
  (void)__builtin_cpu_is("sapphirerapids");
  (void)__builtin_cpu_is("tremont");
  (void)__builtin_cpu_is("westmere");
  (void)__builtin_cpu_is("znver1");
  (void)__builtin_cpu_is("znver2");
  (void)__builtin_cpu_is("znver3");
}

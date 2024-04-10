// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -ffreestanding %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -ffreestanding %s -O3 -triple=i386-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86
// FIXME: This is testing optimized generation of shuffle instructions and should be fixed.


#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions, checking if the masks are correct
//

__m256 x(__m256 a, __m256 b) {
  // CHECK-LABEL: x
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  return _mm256_shuffle_ps(a, b, 203);
}

__m128d test_mm_permute_pd(__m128d a) {
  // CHECK-LABEL: test_mm_permute_pd
  // CHECK: shufflevector{{.*}}<i32 1, i32 0>
  return _mm_permute_pd(a, 1);
}

__m256d test_mm256_permute_pd(__m256d a) {
  // CHECK-LABEL: test_mm256_permute_pd
  // CHECK: shufflevector{{.*}}<i32 1, i32 0, i32 3, i32 2>
  return _mm256_permute_pd(a, 5);
}

__m128 test_mm_permute_ps(__m128 a) {
  // CHECK-LABEL: test_mm_permute_ps
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0>
  return _mm_permute_ps(a, 0x1b);
}

// Test case for PR12401
__m128 test_mm_permute_ps2(__m128 a) {
  // CHECK-LABEL: test_mm_permute_ps2
  // CHECK: shufflevector{{.*}}<i32 2, i32 1, i32 2, i32 3>
  return _mm_permute_ps(a, 0xe6);
}

__m256 test_mm256_permute_ps(__m256 a) {
  // CHECK-LABEL: test_mm256_permute_ps
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  return _mm256_permute_ps(a, 0x1b);
}

__m256d test_mm256_permute2f128_pd(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_permute2f128_pd
  // CHECK: shufflevector{{.*}}<i32 2, i32 3, i32 6, i32 7> 
  return _mm256_permute2f128_pd(a, b, 0x31);
}

__m256 test_mm256_permute2f128_ps(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_permute2f128_ps
  // CHECK: shufflevector{{.*}}<i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_permute2f128_ps(a, b, 0x13);
}

__m256i test_mm256_permute2f128_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_permute2f128_si256
  // X64: shufflevector{{.*}}<i32 0, i32 1, i32 4, i32 5>
  // X86: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_permute2f128_si256(a, b, 0x20);
}

__m128
test_mm_broadcast_ss(float const *__a) {
  // CHECK-LABEL: test_mm_broadcast_ss
  // CHECK: insertelement <4 x float> {{.*}}, i64 0
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
  return _mm_broadcast_ss(__a);
}

__m256d
test_mm256_broadcast_sd(double const *__a) {
  // CHECK-LABEL: test_mm256_broadcast_sd
  // CHECK: insertelement <4 x double> {{.*}}, i64 0
  // CHECK: shufflevector <4 x double> {{.*}}, <4 x double> poison, <4 x i32> zeroinitializer
  return _mm256_broadcast_sd(__a);
}

__m256
test_mm256_broadcast_ss(float const *__a) {
  // CHECK-LABEL: test_mm256_broadcast_ss
  // CHECK: insertelement <8 x float> {{.*}}, i64 0
  // CHECK: shufflevector <8 x float> {{.*}}, <8 x float> poison, <8 x i32> zeroinitializer
  return _mm256_broadcast_ss(__a);
}

// Make sure we have the correct mask for each insertf128 case.

__m256 test_mm256_insertf128_ps_0(__m256 a, __m128 b) {
  // CHECK-LABEL: test_mm256_insertf128_ps_0
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  return _mm256_insertf128_ps(a, b, 0);
}

__m256d test_mm256_insertf128_pd_0(__m256d a, __m128d b) {
  // CHECK-LABEL: test_mm256_insertf128_pd_0
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 6, i32 7>
  return _mm256_insertf128_pd(a, b, 0);
}

__m256i test_mm256_insertf128_si256_0(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_insertf128_si256_0
  // X64: shufflevector{{.*}}<i32 0, i32 1, i32 6, i32 7>
  // X86: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  return _mm256_insertf128_si256(a, b, 0);
}

__m256 test_mm256_insertf128_ps_1(__m256 a, __m128 b) {
  // CHECK-LABEL: test_mm256_insertf128_ps_1
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf128_ps(a, b, 1);
}

__m256d test_mm256_insertf128_pd_1(__m256d a, __m128d b) {
  // CHECK-LABEL: test_mm256_insertf128_pd_1
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 4, i32 5>
  return _mm256_insertf128_pd(a, b, 1);
}

__m256i test_mm256_insertf128_si256_1(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_insertf128_si256_1
  // X64: shufflevector{{.*}}<i32 0, i32 1, i32 4, i32 5>
  // X86: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf128_si256(a, b, 1);
}

// Make sure we have the correct mask for each extractf128 case.

__m128 test_mm256_extractf128_ps_0(__m256 a) {
  // X64-LABEL: test_mm256_extractf128_ps_0
  // X64: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3>
  //
  // X86-LABEL: test_mm256_extractf128_ps_0
  // X86: shufflevector{{.*}}<i32 0, i32 1>
  return _mm256_extractf128_ps(a, 0);
}

__m128d test_mm256_extractf128_pd_0(__m256d a) {
  // CHECK-LABEL: test_mm256_extractf128_pd_0
  // CHECK: shufflevector{{.*}}<i32 0, i32 1>
  return _mm256_extractf128_pd(a, 0);
}

__m128i test_mm256_extractf128_si256_0(__m256i a) {
  // CHECK-LABEL: test_mm256_extractf128_si256_0
  // CHECK: shufflevector{{.*}}<i32 0, i32 1>
  return _mm256_extractf128_si256(a, 0);
}

__m128 test_mm256_extractf128_ps_1(__m256 a) {
  // X64-LABEL: test_mm256_extractf128_ps_1
  // X64: shufflevector{{.*}}<i32 4, i32 5, i32 6, i32 7>
  //
  // X86-LABEL: test_mm256_extractf128_ps_1
  // X86: shufflevector{{.*}}<i32 2, i32 3>
  return _mm256_extractf128_ps(a, 1);
}

__m128d test_mm256_extractf128_pd_1(__m256d a) {
  // CHECK-LABEL: test_mm256_extractf128_pd_1
  // CHECK: shufflevector{{.*}}<i32 2, i32 3>
  return _mm256_extractf128_pd(a, 1);
}

__m128i test_mm256_extractf128_si256_1(__m256i a) {
  // CHECK-LABEL: test_mm256_extractf128_si256_1
  // CHECK: shufflevector{{.*}}<i32 2, i32 3>
  return _mm256_extractf128_si256(a, 1);
}

__m256 test_mm256_set_m128(__m128 hi, __m128 lo) {
  // CHECK-LABEL: test_mm256_set_m128
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_set_m128(hi, lo);
}

__m256d test_mm256_set_m128d(__m128d hi, __m128d lo) {
  // CHECK-LABEL: test_mm256_set_m128d
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3>
  return _mm256_set_m128d(hi, lo);
}

__m256i test_mm256_set_m128i(__m128i hi, __m128i lo) {
  // CHECK-LABEL: test_mm256_set_m128i
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3>  
  return _mm256_set_m128i(hi, lo);
}

__m256 test_mm256_setr_m128(__m128 hi, __m128 lo) {
  // CHECK-LABEL: test_mm256_setr_m128
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_setr_m128(lo, hi);
}

__m256d test_mm256_setr_m128d(__m128d hi, __m128d lo) {
  // CHECK-LABEL: test_mm256_setr_m128d
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3>
  return _mm256_setr_m128d(lo, hi);
}

__m256i test_mm256_setr_m128i(__m128i hi, __m128i lo) {
  // CHECK-LABEL: test_mm256_setr_m128i
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 2, i32 3>
  return _mm256_setr_m128i(lo, hi);
}

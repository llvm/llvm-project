// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_mm_abs_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.pabs.b.128
  return _mm_abs_epi8(a);
}

__m128i test_mm_abs_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pabs.w.128
  return _mm_abs_epi16(a);
}

__m128i test_mm_abs_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.pabs.d.128
  return _mm_abs_epi32(a);
}

__m128i test_mm_alignr_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  return _mm_alignr_epi8(a, b, 2);
}

__m128i test2_mm_alignr_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test2_mm_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  return _mm_alignr_epi8(a, b, 17);
}

__m128i test_mm_hadd_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadd_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.w.128
  return _mm_hadd_epi16(a, b);
}

__m128i test_mm_hadd_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadd_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phadd.d.128
  return _mm_hadd_epi32(a, b);
}

__m128i test_mm_hadds_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadds_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.sw.128
  return _mm_hadds_epi16(a, b);
}

__m128i test_mm_hsub_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsub_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.w.128
  return _mm_hsub_epi16(a, b);
}

__m128i test_mm_hsub_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsub_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phsub.d.128
  return _mm_hsub_epi32(a, b);
}

__m128i test_mm_hsubs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsubs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.sw.128
  return _mm_hsubs_epi16(a, b);
}

__m128i test_mm_maddubs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_maddubs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128
  return _mm_maddubs_epi16(a, b);
}

__m128i test_mm_mulhrs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_mulhrs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128
  return _mm_mulhrs_epi16(a, b);
}

__m128i test_mm_shuffle_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shuffle_epi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.pshuf.b.128
  return _mm_shuffle_epi8(a, b);
}

__m128i test_mm_sign_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.psign.b.128
  return _mm_sign_epi8(a, b);
}

__m128i test_mm_sign_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.psign.w.128
  return _mm_sign_epi16(a, b);
}

__m128i test_mm_sign_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.psign.d.128
  return _mm_sign_epi32(a, b);
}

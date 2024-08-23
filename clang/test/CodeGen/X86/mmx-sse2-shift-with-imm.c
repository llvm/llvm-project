// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature +mmx \
// RUN:  -target-feature +sse2 -O0 -emit-llvm %s -o - | FileCheck %s

// Test that mmx/sse2 shift intrinsics map to the expected builtins.

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m64 check__mm_slli_pi16(__m64 m) {
  // CHECK-LABEL: @check__mm_slli_pi16
  // CHECK: @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_slli_pi16(m, 8);
}

__m64 check__mm_slli_pi32(__m64 m) {
  // CHECK-LABEL: @check__mm_slli_pi32
  // CHECK: @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_slli_pi32(m, 8);
}

__m64 check__mm_slli_si64(__m64 m) {
  // CHECK-LABEL: @check__mm_slli_si64
  // CHECK: @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  return _mm_slli_si64(m, 8);
}

__m64 check__mm_srai_pi16(__m64 m) {
  // CHECK-LABEL: @check__mm_srai_pi16
  // CHECK: @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_srai_pi16(m, 8);
}

__m64 check__mm_srai_pi32(__m64 m) {
  // CHECK-LABEL: @check__mm_srai_pi32
  // CHECK: @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_srai_pi32(m, 8);
}

__m64 check__mm_srli_pi16(__m64 m) {
  // CHECK-LABEL: @check__mm_srli_pi16
  // CHECK: @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_srli_pi16(m, 8);
}

__m64 check__mm_srli_pi32(__m64 m) {
  // CHECK-LABEL: @check__mm_srli_pi32
  // CHECK: @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_srli_pi32(m, 8);
}

__m64 check__mm_srli_si64(__m64 m) {
  // CHECK-LABEL: @check__mm_srli_si64
  // CHECK: @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  return _mm_srli_si64(m, 8);
}

__m128i check__mm_slli_epi16(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_slli_epi16(a, b);
}

__m128i check__mm_slli_epi32(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_slli_epi32
  // CHECK: @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_slli_epi32(a, b);
}

__m128i check__mm_slli_epi64(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_slli_epi64
  // CHECK: @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  return _mm_slli_epi64(a, b);
}

__m128i check__mm_srai_epi16(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_srai_epi16
  // CHECK: @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_srai_epi16(a, b);
}

__m128i check__mm_srai_epi32(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_srai_epi32
  // CHECK: @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_srai_epi32(a, b);
}

__m128i check__mm_srli_epi16(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_srli_epi16
  // CHECK: @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  return _mm_srli_epi16(a, b);
}

__m128i check__mm_srli_epi32(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_srli_epi32
  // CHECK: @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  return _mm_srli_epi32(a, b);
}

__m128i check__mm_srli_epi64(__m128i a, const int b) {
  // CHECK-LABEL: @check__mm_srli_epi64
  // CHECK: @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  return _mm_srli_epi64(a, b);
}

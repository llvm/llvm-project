// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512er -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m512d test_mm512_rsqrt28_round_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_rsqrt28_round_pd
  // CHECK: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_rsqrt28_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_mask_rsqrt28_round_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_rsqrt28_round_pd
  // check: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_mask_rsqrt28_round_pd(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_maskz_rsqrt28_round_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_rsqrt28_round_pd
  // check: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_maskz_rsqrt28_round_pd(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_rsqrt28_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_rsqrt28_pd
  // CHECK: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_rsqrt28_pd(a);
}

__m512d test_mm512_mask_rsqrt28_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_rsqrt28_pd
  // check: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_mask_rsqrt28_pd(s, m, a);
}

__m512d test_mm512_maskz_rsqrt28_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_rsqrt28_pd
  // check: @llvm.x86.avx512.rsqrt28.pd
  return _mm512_maskz_rsqrt28_pd(m, a);
}

__m512 test_mm512_rsqrt28_round_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_rsqrt28_round_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_rsqrt28_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_mask_rsqrt28_round_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_rsqrt28_round_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_mask_rsqrt28_round_ps(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_maskz_rsqrt28_round_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_rsqrt28_round_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_maskz_rsqrt28_round_ps(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_rsqrt28_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_rsqrt28_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_rsqrt28_ps(a);
}

__m512 test_mm512_mask_rsqrt28_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_rsqrt28_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_mask_rsqrt28_ps(s, m, a);
}

__m512 test_mm512_maskz_rsqrt28_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_rsqrt28_ps
  // CHECK: @llvm.x86.avx512.rsqrt28.ps
  return _mm512_maskz_rsqrt28_ps(m, a);
}

__m128 test_mm_rsqrt28_round_ss(__m128 a, __m128 b) {
  // check-label: @test_mm_rsqrt28_round_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_rsqrt28_round_ss(a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_mask_rsqrt28_round_ss(__m128 s, __mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_mask_rsqrt28_round_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_mask_rsqrt28_round_ss(s, m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_maskz_rsqrt28_round_ss(__mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_maskz_rsqrt28_round_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_maskz_rsqrt28_round_ss(m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_rsqrt28_ss(__m128 a, __m128 b) {
  // check-label: @test_mm_rsqrt28_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_rsqrt28_ss(a, b);
}

__m128 test_mm_mask_rsqrt28_ss(__m128 s, __mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_mask_rsqrt28_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_mask_rsqrt28_ss(s, m, a, b);
}

__m128 test_mm_maskz_rsqrt28_ss(__mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_maskz_rsqrt28_ss
  // check: @llvm.x86.avx512.rsqrt28.ss
  return _mm_maskz_rsqrt28_ss(m, a, b);
}

__m128d test_mm_rsqrt28_round_sd(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_rsqrt28_round_sd
  // CHECK: @llvm.x86.avx512.rsqrt28.sd
  return _mm_rsqrt28_round_sd(a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128d test_mm_mask_rsqrt28_round_sd(__m128d s, __mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_rsqrt28_round_sd
  // CHECK: @llvm.x86.avx512.rsqrt28.sd
  return _mm_mask_rsqrt28_round_sd(s, m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128d test_mm_maskz_rsqrt28_round_sd(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt28_round_sd
  // CHECK: @llvm.x86.avx512.rsqrt28.sd
  return _mm_maskz_rsqrt28_round_sd(m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_rcp28_round_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_rcp28_round_pd
  // CHECK: @llvm.x86.avx512.rcp28.pd
  return _mm512_rcp28_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_mask_rcp28_round_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_rcp28_round_pd
  // check: @llvm.x86.avx512.rcp28.pd
  return _mm512_mask_rcp28_round_pd(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_maskz_rcp28_round_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_rcp28_round_pd
  // check: @llvm.x86.avx512.rcp28.pd
  return _mm512_maskz_rcp28_round_pd(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_rcp28_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_rcp28_pd
  // CHECK: @llvm.x86.avx512.rcp28.pd
  return _mm512_rcp28_pd(a);
}

__m512d test_mm512_mask_rcp28_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_rcp28_pd
  // check: @llvm.x86.avx512.rcp28.pd
  return _mm512_mask_rcp28_pd(s, m, a);
}

__m512d test_mm512_maskz_rcp28_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_rcp28_pd
  // check: @llvm.x86.avx512.rcp28.pd
  return _mm512_maskz_rcp28_pd(m, a);
}

__m512 test_mm512_rcp28_round_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_rcp28_round_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_rcp28_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_mask_rcp28_round_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_rcp28_round_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_mask_rcp28_round_ps(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_maskz_rcp28_round_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_rcp28_round_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_maskz_rcp28_round_ps(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_rcp28_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_rcp28_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_rcp28_ps(a);
}

__m512 test_mm512_mask_rcp28_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_rcp28_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_mask_rcp28_ps(s, m, a);
}

__m512 test_mm512_maskz_rcp28_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_rcp28_ps
  // CHECK: @llvm.x86.avx512.rcp28.ps
  return _mm512_maskz_rcp28_ps(m, a);
}

__m128 test_mm_rcp28_round_ss(__m128 a, __m128 b) {
  // check-label: @test_mm_rcp28_round_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_rcp28_round_ss(a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_mask_rcp28_round_ss(__m128 s, __mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_mask_rcp28_round_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_mask_rcp28_round_ss(s, m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_maskz_rcp28_round_ss(__mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_maskz_rcp28_round_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_maskz_rcp28_round_ss(m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128 test_mm_rcp28_ss(__m128 a, __m128 b) {
  // check-label: @test_mm_rcp28_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_rcp28_ss(a, b);
}

__m128 test_mm_mask_rcp28_ss(__m128 s, __mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_mask_rcp28_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_mask_rcp28_ss(s, m, a, b);
}

__m128 test_mm_maskz_rcp28_ss(__mmask16 m, __m128 a, __m128 b) {
  // check-label: @test_mm_maskz_rcp28_ss
  // check: @llvm.x86.avx512.rcp28.ss
  return _mm_maskz_rcp28_ss(m, a, b);
}

__m128d test_mm_rcp28_round_sd(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_rcp28_round_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_rcp28_round_sd(a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128d test_mm_mask_rcp28_round_sd(__m128d s, __mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_rcp28_round_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_mask_rcp28_round_sd(s, m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128d test_mm_maskz_rcp28_round_sd(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_maskz_rcp28_round_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_maskz_rcp28_round_sd(m, a, b, _MM_FROUND_TO_NEAREST_INT);
}

__m128d test_mm_rcp28_sd(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_rcp28_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_rcp28_sd(a, b);
}

__m128d test_mm_mask_rcp28_sd(__m128d s, __mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_rcp28_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_mask_rcp28_sd(s, m, a, b);
}

__m128d test_mm_maskz_rcp28_sd(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_maskz_rcp28_sd
  // CHECK: @llvm.x86.avx512.rcp28.sd
  return _mm_maskz_rcp28_sd(m, a, b);
}

__m512d test_mm512_exp2a23_round_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_exp2a23_round_pd
  // CHECK: @llvm.x86.avx512.exp2.pd
  return _mm512_exp2a23_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_mask_exp2a23_round_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_exp2a23_round_pd
  // check: @llvm.x86.avx512.exp2.pd
  return _mm512_mask_exp2a23_round_pd(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_maskz_exp2a23_round_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_exp2a23_round_pd
  // check: @llvm.x86.avx512.exp2.pd
  return _mm512_maskz_exp2a23_round_pd(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_exp2a23_pd(__m512d a) {
  // CHECK-LABEL: @test_mm512_exp2a23_pd
  // CHECK: @llvm.x86.avx512.exp2.pd
  return _mm512_exp2a23_pd(a);
}

__m512d test_mm512_mask_exp2a23_pd(__m512d s, __mmask8 m, __m512d a) {
  // check-label: @test_mm512_mask_exp2a23_pd
  // check: @llvm.x86.avx512.exp2.pd
  return _mm512_mask_exp2a23_pd(s, m, a);
}

__m512d test_mm512_maskz_exp2a23_pd(__mmask8 m, __m512d a) {
  // check-label: @test_mm512_maskz_exp2a23_pd
  // check: @llvm.x86.avx512.exp2.pd
  return _mm512_maskz_exp2a23_pd(m, a);
}

__m512 test_mm512_exp2a23_round_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_exp2a23_round_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_exp2a23_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_mask_exp2a23_round_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_exp2a23_round_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_mask_exp2a23_round_ps(s, m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_maskz_exp2a23_round_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_exp2a23_round_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_maskz_exp2a23_round_ps(m, a, _MM_FROUND_TO_NEAREST_INT);
}

__m512 test_mm512_exp2a23_ps(__m512 a) {
  // CHECK-LABEL: @test_mm512_exp2a23_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_exp2a23_ps(a);
}

__m512 test_mm512_mask_exp2a23_ps(__m512 s, __mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_mask_exp2a23_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_mask_exp2a23_ps(s, m, a);
}

__m512 test_mm512_maskz_exp2a23_ps(__mmask16 m, __m512 a) {
  // CHECK-LABEL: @test_mm512_maskz_exp2a23_ps
  // CHECK: @llvm.x86.avx512.exp2.ps
  return _mm512_maskz_exp2a23_ps(m, a);
}


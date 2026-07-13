// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X64

#include <immintrin.h>
#include <stddef.h>

__m256i test_mm512_cvtts_pd_epi32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_pd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_cvtts_pd_epi32(A);
}

__m256i test_mm512_mask_cvtts_pd_epi32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_pd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_mask_cvtts_pd_epi32(W, U, A);
}

__m256i test_mm512_maskz_cvtts_pd_epi32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_pd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_pd_epi32(U, A);
}

__m256i test_mm512_cvtts_roundpd_epi32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epi32(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtts_roundpd_epi32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epi32(W, U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtts_roundpd_epi32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epi32(U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvtts_pd_epu32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_pd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_cvtts_pd_epu32(A);
}

__m256i test_mm512_mask_cvtts_pd_epu32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_pd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_mask_cvtts_pd_epu32(W, U, A);
}

__m256i test_mm512_maskz_cvtts_pd_epu32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_pd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_pd_epu32(U, A);
}

__m256i test_mm512_cvtts_roundpd_epu32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epu32(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtts_roundpd_epu32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epu32(W, U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtts_roundpd_epu32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epu32(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtts_ps_epi32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_ps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_cvtts_ps_epi32(A);
}

__m512i test_mm512_mask_cvtts_ps_epi32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_ps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_mask_cvtts_ps_epi32(W, U, A);
}

__m512i test_mm512_maskz_cvtts_ps_epi32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_ps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_ps_epi32(U, A);
}

__m512i test_mm512_cvtts_roundps_epi32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_cvtts_roundps_epi32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epi32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_mask_cvtts_roundps_epi32(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundps_epi32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_roundps_epi32(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtts_ps_epu32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_ps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_cvtts_ps_epu32(A);
}

__m512i test_mm512_mask_cvtts_ps_epu32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_ps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_mask_cvtts_ps_epu32(W, U, A);
}

__m512i test_mm512_maskz_cvtts_ps_epu32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_ps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_ps_epu32(U, A);
}

__m512i test_mm512_cvtts_roundps_epu32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_cvtts_roundps_epu32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epu32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_mask_cvtts_roundps_epu32(W, U, A, _MM_FROUND_NO_EXC);
}
__m512i test_mm512_maskz_cvtts_roundps_epu32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_roundps_epu32(U, A, _MM_FROUND_NO_EXC);
}

// X64: {{.*}}
// X86: {{.*}}

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-unknown-unknown -target-feature +avx10.2-512 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

int test_mm_cvttssd_i32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_i32
  // CHECK: @llvm.x86.avx512.vcvttssd2si
  return _mm_cvtts_roundsd_i32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttssd_si32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_si32(
  // CHECK: @llvm.x86.avx512.vcvttssd2si(<2 x double>
  return _mm_cvtts_roundsd_si32(__A, _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvttssd_u32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_u32(
  // CHECK: @llvm.x86.avx512.vcvttssd2usi(<2 x double>
  return _mm_cvtts_roundsd_u32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttsss_i32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_i32(
  // CHECK: @llvm.x86.avx512.vcvttsss2si(<4 x float>
  return _mm_cvtts_roundss_i32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttsss_si32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_si32(
  // CHECK: @llvm.x86.avx512.vcvttsss2si(<4 x float>
  return _mm_cvtts_roundss_si32(__A, _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvttsss_u32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_u32(
  // CHECK: @llvm.x86.avx512.vcvttsss2usi(<4 x float>
  return _mm_cvtts_roundss_u32(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvttspd_epi32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvttspd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_cvttspd_epi32(A);
}

__m256i test_mm512_mask_cvttspd_epi32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvttspd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_mask_cvttspd_epi32(W, U, A);
}

__m256i test_mm512_maskz_cvttspd_epi32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvttspd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_maskz_cvttspd_epi32(U, A);
}

__m256i test_mm512_cvtts_roundpd_epi32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epi32(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtts_roundpd_epi32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epi32(W, U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtts_roundpd_epi32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2dqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epi32(U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvttspd_epu32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvttspd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_cvttspd_epu32(A);
}

__m256i test_mm512_mask_cvttspd_epu32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvttspd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_mask_cvttspd_epu32(W, U, A);
}

__m256i test_mm512_maskz_cvttspd_epu32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvttspd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_maskz_cvttspd_epu32(U, A);
}

__m256i test_mm512_cvtts_roundpd_epu32(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epu32(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtts_roundpd_epu32(__m256i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epu32(W, U, A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtts_roundpd_epu32(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttpd2udqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epu32(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttsps_epi32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvttsps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_cvttsps_epi32(A);
}

__m512i test_mm512_mask_cvttsps_epi32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvttsps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_mask_cvttsps_epi32(W, U, A);
}

__m512i test_mm512_maskz_cvttsps_epi32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvttsps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_maskz_cvttsps_epi32(U, A);
}

__m512i test_mm512_cvtts_roundps_epi32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_cvtts_roundps_epi32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epi32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_mask_cvtts_roundps_epi32(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundps_epi32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2dqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_roundps_epi32(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttsps_epu32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvttsps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_cvttsps_epu32(A);
}

__m512i test_mm512_mask_cvttsps_epu32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvttsps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_mask_cvttsps_epu32(W, U, A);
}

__m512i test_mm512_maskz_cvttsps_epu32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvttsps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_maskz_cvttsps_epu32(U, A);
}

__m512i test_mm512_cvtts_roundps_epu32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_cvtts_roundps_epu32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epu32(__m512i W, __mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_mask_cvtts_roundps_epu32(W, U, A, _MM_FROUND_NO_EXC);
}
__m512i test_mm512_maskz_cvtts_roundps_epu32(__mmask8 U, __m512 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.vcvttps2udqs.round.512(<16 x float>
  return _mm512_maskz_cvtts_roundps_epu32(U, A, _MM_FROUND_NO_EXC);
}

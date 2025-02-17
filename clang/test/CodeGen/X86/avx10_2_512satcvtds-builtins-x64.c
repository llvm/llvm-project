// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10.2-512 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

long long test_mm_cvttssd_si64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_si64(
  // CHECK: @llvm.x86.avx10.vcvttsd2sis64(<2 x double>
  return _mm_cvtts_roundsd_si64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvttssd_i64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_i64(
  // CHECK: @llvm.x86.avx10.vcvttsd2sis64(<2 x double>
  return _mm_cvtts_roundsd_i64(__A, _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvttssd_u64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttssd_u64(
  // CHECK: @llvm.x86.avx10.vcvttsd2usis64(<2 x double>
  return _mm_cvtts_roundsd_u64(__A, _MM_FROUND_NO_EXC);
}

float test_mm_cvttsss_i64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_i64(
  // CHECK: @llvm.x86.avx10.vcvttss2sis64(<4 x float>
  return _mm_cvtts_roundss_i64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvttsss_si64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_si64(
  // CHECK: @llvm.x86.avx10.vcvttss2sis64(<4 x float>
  return _mm_cvtts_roundss_si64(__A, _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvttsss_u64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttsss_u64(
  // CHECK: @llvm.x86.avx10.vcvttss2usis64(<4 x float>
  return _mm_cvtts_roundss_u64(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttspd_epi64(__m512d A) {
  // CHECK-LABEL: test_mm512_cvttspd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_cvttspd_epi64(A);
}

__m512i test_mm512_mask_cvttspd_epi64(__m512i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvttspd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_mask_cvttspd_epi64(W, U, A);
}

__m512i test_mm512_maskz_cvttspd_epi64(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvttspd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_maskz_cvttspd_epi64(U, A);
}

__m512i test_mm512_cvtts_roundpd_epi64(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epi64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundpd_epi64(__m512i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epi64(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundpd_epi64(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2qqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epi64(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttspd_epu64(__m512d A) {
  // CHECK-LABEL: test_mm512_cvttspd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_cvttspd_epu64(A);
}

__m512i test_mm512_mask_cvttspd_epu64(__m512i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvttspd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_mask_cvttspd_epu64(W, U, A);
}

__m512i test_mm512_maskz_cvttspd_epu64(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvttspd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_maskz_cvttspd_epu64(U, A);
}

__m512i test_mm512_cvtts_roundpd_epu64(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtts_roundpd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_cvtts_roundpd_epu64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundpd_epu64(__m512i W, __mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundpd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_mask_cvtts_roundpd_epu64(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundpd_epu64(__mmask8 U, __m512d A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundpd_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttpd2uqqs.round.512(<8 x double>
  return _mm512_maskz_cvtts_roundpd_epu64(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttsps_epi64(__m256 A) {
  // CHECK-LABEL: test_mm512_cvttsps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_cvttsps_epi64(A);
}

__m512i test_mm512_mask_cvttsps_epi64(__m512i W, __mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_mask_cvttsps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_mask_cvttsps_epi64(W, U, A);
}

__m512i test_mm512_maskz_cvttsps_epi64(__mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_maskz_cvttsps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_maskz_cvttsps_epi64(U, A);
}

__m512i test_mm512_cvtts_roundps_epi64(__m256 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_cvtts_roundps_epi64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epi64(__m512i W, __mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_mask_cvtts_roundps_epi64(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundps_epi64(__mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epi64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2qqs.round.512(<8 x float>
  return _mm512_maskz_cvtts_roundps_epi64(U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttsps_epu64(__m256 A) {
  // CHECK-LABEL: test_mm512_cvttsps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_cvttsps_epu64(A);
}

__m512i test_mm512_mask_cvttsps_epu64(__m512i W, __mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_mask_cvttsps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_mask_cvttsps_epu64(W, U, A);
}

__m512i test_mm512_maskz_cvttsps_epu64(__mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_maskz_cvttsps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_maskz_cvttsps_epu64(U, A);
}

__m512i test_mm512_cvtts_roundps_epu64(__m256 A) {
  // CHECK-LABEL: test_mm512_cvtts_roundps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_cvtts_roundps_epu64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtts_roundps_epu64(__m512i W, __mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_mask_cvtts_roundps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_mask_cvtts_roundps_epu64(W, U, A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtts_roundps_epu64(__mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm512_maskz_cvtts_roundps_epu64
  // CHECK: @llvm.x86.avx10.mask.vcvttps2uqqs.round.512(<8 x float>
  return _mm512_maskz_cvtts_roundps_epu64(U, A, _MM_FROUND_NO_EXC);
}

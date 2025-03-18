// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512bh test_mm512_minmax_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_minmax_pbh(
  // CHECK: call <32 x bfloat> @llvm.x86.avx10.vminmaxbf16512(
  return _mm512_minmax_pbh(__A, __B, 127);
}

__m512bh test_mm512_mask_minmax_pbh(__m512bh __A, __mmask32 __B, __m512bh __C, __m512bh __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_pbh(
  // CHECK: call <32 x bfloat> @llvm.x86.avx10.vminmaxbf16512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_minmax_pbh(__A, __B, __C, __D, 127);
}

__m512bh test_mm512_maskz_minmax_pbh(__mmask32 __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_pbh(
  // CHECK: call <32 x bfloat> @llvm.x86.avx10.vminmaxbf16512(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_minmax_pbh(__A, __B, __C, 127);
}

__m512d test_mm512_minmax_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_minmax_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_minmax_pd(__A, __B, 127);
}

__m512d test_mm512_mask_minmax_pd(__m512d __A, __mmask8 __B, __m512d __C, __m512d __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_mask_minmax_pd(__A, __B, __C, __D, 127);
}

__m512d test_mm512_maskz_minmax_pd(__mmask8 __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_maskz_minmax_pd(__A, __B, __C, 127);
}

__m512d test_mm512_minmax_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_minmax_round_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_minmax_round_pd(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_minmax_round_pd(__m512d __A, __mmask8 __B, __m512d __C, __m512d __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_round_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_mask_minmax_round_pd(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_minmax_round_pd(__mmask8 __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_round_pd(
  // CHECK: call <8 x double> @llvm.x86.avx10.mask.vminmaxpd.round(
  return _mm512_maskz_minmax_round_pd(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_minmax_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_minmax_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_minmax_ph(__A, __B, 127);
}

__m512h test_mm512_mask_minmax_ph(__m512h __A, __mmask32 __B, __m512h __C, __m512h __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_mask_minmax_ph(__A, __B, __C, __D, 127);
}

__m512h test_mm512_maskz_minmax_ph(__mmask32 __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_maskz_minmax_ph(__A, __B, __C, 127);
}

__m512h test_mm512_minmax_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_minmax_round_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_minmax_round_ph(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_minmax_round_ph(__m512h __A, __mmask32 __B, __m512h __C, __m512h __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_round_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_mask_minmax_round_ph(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_minmax_round_ph(__mmask32 __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_round_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vminmaxph.round(
  return _mm512_maskz_minmax_round_ph(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_minmax_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_minmax_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_minmax_ps(__A, __B, 127);
}

__m512 test_mm512_mask_minmax_ps(__m512 __A, __mmask16 __B, __m512 __C, __m512 __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_mask_minmax_ps(__A, __B, __C, __D, 127);
}

__m512 test_mm512_maskz_minmax_ps(__mmask16 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_maskz_minmax_ps(__A, __B, __C, 127);
}

__m512 test_mm512_minmax_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_minmax_round_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_minmax_round_ps(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_minmax_round_ps(__m512 __A, __mmask16 __B, __m512 __C, __m512 __D) {
  // CHECK-LABEL: @test_mm512_mask_minmax_round_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_mask_minmax_round_ps(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_minmax_round_ps(__mmask16 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_minmax_round_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vminmaxps.round(
  return _mm512_maskz_minmax_round_ps(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

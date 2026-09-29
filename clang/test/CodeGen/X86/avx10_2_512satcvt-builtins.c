// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2 \
// RUN: -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2 \
// RUN: -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_ipcvts_bf16_epi8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs512
  return _mm512_ipcvts_bf16_epi8(__A);
}

__m512i test_mm512_mask_ipcvts_bf16_epi8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvts_bf16_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_bf16_epi8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_bf16_epi8
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs512
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvts_bf16_epi8(__A, __B);
}

__m512i test_mm512_ipcvts_bf16_epu8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs512
  return _mm512_ipcvts_bf16_epu8(__A);
}

__m512i test_mm512_mask_ipcvts_bf16_epu8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvts_bf16_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_bf16_epu8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_bf16_epu8
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs512
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvts_bf16_epu8(__A, __B);
}

__m512i test_mm512_ipcvts_ph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_ipcvts_ph_epi8(__A);
}

__m512i test_mm512_mask_ipcvts_ph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_mask_ipcvts_ph_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_ph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_maskz_ipcvts_ph_epi8(__A, __B);
}

__m512i test_mm512_ipcvts_roundph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_ipcvts_roundph_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvts_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_mask_ipcvts_roundph_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvts_roundph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_maskz_ipcvts_roundph_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvts_ph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_ipcvts_ph_epu8(__A);
}

__m512i test_mm512_mask_ipcvts_ph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_mask_ipcvts_ph_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_ph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_maskz_ipcvts_ph_epu8(__A, __B);
}

__m512i test_mm512_ipcvts_roundph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_ipcvts_roundph_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvts_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_mask_ipcvts_roundph_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvts_roundph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_maskz_ipcvts_roundph_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvts_ps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_ipcvts_ps_epi8(__A);
}

__m512i test_mm512_mask_ipcvts_ps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_mask_ipcvts_ps_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_ps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_maskz_ipcvts_ps_epi8(__A, __B);
}

__m512i test_mm512_ipcvts_roundps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_ipcvts_roundps_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvts_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_mask_ipcvts_roundps_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvts_roundps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_maskz_ipcvts_roundps_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvts_ps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_ipcvts_ps_epu8(__A);
}

__m512i test_mm512_mask_ipcvts_ps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_mask_ipcvts_ps_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvts_ps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_maskz_ipcvts_ps_epu8(__A, __B);
}

__m512i test_mm512_ipcvts_roundps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvts_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_ipcvts_roundps_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvts_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvts_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_mask_ipcvts_roundps_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvts_roundps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvts_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_maskz_ipcvts_roundps_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtts_bf16_epi8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs512(
  return _mm512_ipcvtts_bf16_epi8(__A);
}

__m512i test_mm512_mask_ipcvtts_bf16_epi8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvtts_bf16_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_bf16_epi8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_bf16_epi8
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs512(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvtts_bf16_epi8(__A, __B);
}

__m512i test_mm512_ipcvtts_bf16_epu8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs512(
  return _mm512_ipcvtts_bf16_epu8(__A);
}

__m512i test_mm512_mask_ipcvtts_bf16_epu8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvtts_bf16_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_bf16_epu8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_bf16_epu8
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs512(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvtts_bf16_epu8(__A, __B);
}

__m512i test_mm512_ipcvtts_ph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_ipcvtts_ph_epi8(__A);
}

__m512i test_mm512_mask_ipcvtts_ph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_mask_ipcvtts_ph_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_ph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_ph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_maskz_ipcvtts_ph_epi8(__A, __B);
}

__m512i test_mm512_ipcvtts_roundph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_ipcvtts_roundph_epi8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtts_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_mask_ipcvtts_roundph_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtts_roundph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_maskz_ipcvtts_roundph_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtts_ph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_ipcvtts_ph_epu8(__A);
}

__m512i test_mm512_mask_ipcvtts_ph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_mask_ipcvtts_ph_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_ph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_ph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_maskz_ipcvtts_ph_epu8(__A, __B);
}

__m512i test_mm512_ipcvtts_roundph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_ipcvtts_roundph_epu8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtts_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_mask_ipcvtts_roundph_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtts_roundph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_maskz_ipcvtts_roundph_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtts_ps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_ipcvtts_ps_epi8(__A);
}

__m512i test_mm512_mask_ipcvtts_ps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_mask_ipcvtts_ps_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_ps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_ps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_maskz_ipcvtts_ps_epi8(__A, __B);
}

__m512i test_mm512_ipcvtts_roundps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_ipcvtts_roundps_epi8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtts_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_mask_ipcvtts_roundps_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}


__m512i test_mm512_maskz_ipcvtts_roundps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_maskz_ipcvtts_roundps_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtts_ps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_ipcvtts_ps_epu8(__A);
}

__m512i test_mm512_mask_ipcvtts_ps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_mask_ipcvtts_ps_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtts_ps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_ps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_maskz_ipcvtts_ps_epu8(__A, __B);
}

__m512i test_mm512_ipcvtts_roundps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtts_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_ipcvtts_roundps_epu8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtts_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtts_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_mask_ipcvtts_roundps_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtts_roundps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtts_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_maskz_ipcvtts_roundps_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

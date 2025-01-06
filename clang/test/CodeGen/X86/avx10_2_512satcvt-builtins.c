// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_ipcvtnebf16_epi8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs512
  return _mm512_ipcvtnebf16_epi8(__A);
}

__m512i test_mm512_mask_ipcvtnebf16_epi8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvtnebf16_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtnebf16_epi8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtnebf16_epi8
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs512
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvtnebf16_epi8(__A, __B);
}

__m512i test_mm512_ipcvtnebf16_epu8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs512
  return _mm512_ipcvtnebf16_epu8(__A);
}

__m512i test_mm512_mask_ipcvtnebf16_epu8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvtnebf16_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtnebf16_epu8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtnebf16_epu8
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs512
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvtnebf16_epu8(__A, __B);
}

__m512i test_mm512_ipcvtph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_ipcvtph_epi8(__A);
}

__m512i test_mm512_mask_ipcvtph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_mask_ipcvtph_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_maskz_ipcvtph_epi8(__A, __B);
}

__m512i test_mm512_ipcvt_roundph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_ipcvt_roundph_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvt_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvt_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_mask_ipcvt_roundph_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvt_roundph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvt_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs512
  return _mm512_maskz_ipcvt_roundph_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_ipcvtph_epu8(__A);
}

__m512i test_mm512_mask_ipcvtph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_mask_ipcvtph_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_maskz_ipcvtph_epu8(__A, __B);
}

__m512i test_mm512_ipcvt_roundph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_ipcvt_roundph_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvt_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvt_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_mask_ipcvt_roundph_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvt_roundph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvt_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs512
  return _mm512_maskz_ipcvt_roundph_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_ipcvtps_epi8(__A);
}

__m512i test_mm512_mask_ipcvtps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_mask_ipcvtps_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_maskz_ipcvtps_epi8(__A, __B);
}

__m512i test_mm512_ipcvt_roundps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_ipcvt_roundps_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvt_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvt_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_mask_ipcvt_roundps_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvt_roundps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvt_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs512
  return _mm512_maskz_ipcvt_roundps_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvtps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_ipcvtps_epu8(__A);
}

__m512i test_mm512_mask_ipcvtps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_mask_ipcvtps_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvtps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_maskz_ipcvtps_epu8(__A, __B);
}

__m512i test_mm512_ipcvt_roundps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_ipcvt_roundps_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvt_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvt_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_mask_ipcvt_roundps_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvt_roundps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvt_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs512
  return _mm512_maskz_ipcvt_roundps_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvttnebf16_epi8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs512(
  return _mm512_ipcvttnebf16_epi8(__A);
}

__m512i test_mm512_mask_ipcvttnebf16_epi8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvttnebf16_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttnebf16_epi8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttnebf16_epi8
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs512(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvttnebf16_epi8(__A, __B);
}

__m512i test_mm512_ipcvttnebf16_epu8(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs512(
  return _mm512_ipcvttnebf16_epu8(__A);
}

__m512i test_mm512_mask_ipcvttnebf16_epu8(__m512i __S, __mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_ipcvttnebf16_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttnebf16_epu8(__mmask32 __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttnebf16_epu8
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs512(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_ipcvttnebf16_epu8(__A, __B);
}

__m512i test_mm512_ipcvttph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_ipcvttph_epi8(__A);
}

__m512i test_mm512_mask_ipcvttph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_mask_ipcvttph_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_maskz_ipcvttph_epi8(__A, __B);
}

__m512i test_mm512_ipcvtt_roundph_epi8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtt_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_ipcvtt_roundph_epi8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtt_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtt_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_mask_ipcvtt_roundph_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtt_roundph_epi8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtt_roundph_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs512
  return _mm512_maskz_ipcvtt_roundph_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvttph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_ipcvttph_epu8(__A);
}

__m512i test_mm512_mask_ipcvttph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_mask_ipcvttph_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_maskz_ipcvttph_epu8(__A, __B);
}

__m512i test_mm512_ipcvtt_roundph_epu8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_ipcvtt_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_ipcvtt_roundph_epu8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtt_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtt_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_mask_ipcvtt_roundph_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtt_roundph_epu8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtt_roundph_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs512
  return _mm512_maskz_ipcvtt_roundph_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvttps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_ipcvttps_epi8(__A);
}

__m512i test_mm512_mask_ipcvttps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_mask_ipcvttps_epi8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_maskz_ipcvttps_epi8(__A, __B);
}

__m512i test_mm512_ipcvtt_roundps_epi8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtt_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_ipcvtt_roundps_epi8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtt_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtt_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_mask_ipcvtt_roundps_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}


__m512i test_mm512_maskz_ipcvtt_roundps_epi8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtt_roundps_epi8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs512
  return _mm512_maskz_ipcvtt_roundps_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_ipcvttps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_ipcvttps_epu8(__A);
}

__m512i test_mm512_mask_ipcvttps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_mask_ipcvttps_epu8(__S, __A, __B);
}

__m512i test_mm512_maskz_ipcvttps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvttps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_maskz_ipcvttps_epu8(__A, __B);
}

__m512i test_mm512_ipcvtt_roundps_epu8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_ipcvtt_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_ipcvtt_roundps_epu8(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_ipcvtt_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_ipcvtt_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_mask_ipcvtt_roundps_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_ipcvtt_roundps_epu8(__mmask16 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_ipcvtt_roundps_epu8
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs512
  return _mm512_maskz_ipcvtt_roundps_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

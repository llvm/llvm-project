// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2 \
// RUN: -Wno-invalid-feature-combination -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2 \
// RUN: -Wno-invalid-feature-combination -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_ipcvts_bf16_epi8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs128
  return _mm_ipcvts_bf16_epi8(__A);
}

__m128i test_mm_mask_ipcvts_bf16_epi8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvts_bf16_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_bf16_epi8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvts_bf16_epi8(__A, __B);
}

__m256i test_mm256_ipcvts_bf16_epi8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs256
  return _mm256_ipcvts_bf16_epi8(__A);
}

__m256i test_mm256_mask_ipcvts_bf16_epi8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvts_bf16_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_bf16_epi8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtbf162ibs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvts_bf16_epi8(__A, __B);
}

__m128i test_mm_ipcvts_bf16_epu8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs128
  return _mm_ipcvts_bf16_epu8(__A);
}

__m128i test_mm_mask_ipcvts_bf16_epu8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvts_bf16_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_bf16_epu8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvts_bf16_epu8(__A, __B);
}

__m256i test_mm256_ipcvts_bf16_epu8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs256
  return _mm256_ipcvts_bf16_epu8(__A);
}

__m256i test_mm256_mask_ipcvts_bf16_epu8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvts_bf16_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_bf16_epu8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtbf162iubs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvts_bf16_epu8(__A, __B);
}

__m128i test_mm_ipcvts_ph_epi8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_ipcvts_ph_epi8(__A);
}

__m128i test_mm_mask_ipcvts_ph_epi8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_mask_ipcvts_ph_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_ph_epi8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_maskz_ipcvts_ph_epi8(__A, __B);
}

__m256i test_mm256_ipcvts_ph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_ipcvts_ph_epi8(__A);
}

__m256i test_mm256_mask_ipcvts_ph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_mask_ipcvts_ph_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_ph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_maskz_ipcvts_ph_epi8(__A, __B);
}

__m128i test_mm_ipcvts_ph_epu8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_ipcvts_ph_epu8(__A);
}

__m128i test_mm_mask_ipcvts_ph_epu8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_mask_ipcvts_ph_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_ph_epu8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_maskz_ipcvts_ph_epu8(__A, __B);
}

__m256i test_mm256_ipcvts_ph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_ipcvts_ph_epu8(__A);
}

__m256i test_mm256_mask_ipcvts_ph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_mask_ipcvts_ph_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_ph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_maskz_ipcvts_ph_epu8(__A, __B);
}

__m128i test_mm_ipcvts_ps_epi8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_ipcvts_ps_epi8(__A);
}

__m128i test_mm_mask_ipcvts_ps_epi8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_mask_ipcvts_ps_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_ps_epi8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_maskz_ipcvts_ps_epi8(__A, __B);
}

__m256i test_mm256_ipcvts_ps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_ipcvts_ps_epi8(__A);
}

__m256i test_mm256_mask_ipcvts_ps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_mask_ipcvts_ps_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_ps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_maskz_ipcvts_ps_epi8(__A, __B);
}

__m128i test_mm_ipcvts_ps_epu8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_ipcvts_ps_epu8(__A);
}

__m128i test_mm_mask_ipcvts_ps_epu8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_mask_ipcvts_ps_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvts_ps_epu8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_maskz_ipcvts_ps_epu8(__A, __B);
}

__m256i test_mm256_ipcvts_ps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_ipcvts_ps_epu8(__A);
}

__m256i test_mm256_mask_ipcvts_ps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_mask_ipcvts_ps_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvts_ps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_maskz_ipcvts_ps_epu8(__A, __B);
}

__m128i test_mm_ipcvtts_bf16_epi8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs128
  return _mm_ipcvtts_bf16_epi8(__A);
}

__m128i test_mm_mask_ipcvtts_bf16_epi8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvtts_bf16_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_bf16_epi8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvtts_bf16_epi8(__A, __B);
}

__m256i test_mm256_ipcvtts_bf16_epi8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs256
  return _mm256_ipcvtts_bf16_epi8(__A);
}

__m256i test_mm256_mask_ipcvtts_bf16_epi8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvtts_bf16_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_bf16_epi8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_bf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttbf162ibs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvtts_bf16_epi8(__A, __B);
}

__m128i test_mm_ipcvtts_bf16_epu8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs128
  return _mm_ipcvtts_bf16_epu8(__A);
}

__m128i test_mm_mask_ipcvtts_bf16_epu8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvtts_bf16_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_bf16_epu8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvtts_bf16_epu8(__A, __B);
}

__m256i test_mm256_ipcvtts_bf16_epu8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs256
  return _mm256_ipcvtts_bf16_epu8(__A);
}

__m256i test_mm256_mask_ipcvtts_bf16_epu8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvtts_bf16_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_bf16_epu8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_bf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttbf162iubs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvtts_bf16_epu8(__A, __B);
}

__m128i test_mm_ipcvtts_ph_epi8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_ipcvtts_ph_epi8(__A);
}

__m128i test_mm_mask_ipcvtts_ph_epi8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_mask_ipcvtts_ph_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_ph_epi8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_maskz_ipcvtts_ph_epi8(__A, __B);
}

__m256i test_mm256_ipcvtts_ph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_ipcvtts_ph_epi8(__A);
}

__m256i test_mm256_mask_ipcvtts_ph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_mask_ipcvtts_ph_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_ph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_ph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_maskz_ipcvtts_ph_epi8(__A, __B);
}

__m128i test_mm_ipcvtts_ph_epu8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_ipcvtts_ph_epu8(__A);
}

__m128i test_mm_mask_ipcvtts_ph_epu8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_mask_ipcvtts_ph_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_ph_epu8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_maskz_ipcvtts_ph_epu8(__A, __B);
}

__m256i test_mm256_ipcvtts_ph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_ipcvtts_ph_epu8(__A);
}

__m256i test_mm256_mask_ipcvtts_ph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_mask_ipcvtts_ph_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_ph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_ph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_maskz_ipcvtts_ph_epu8(__A, __B);
}

__m128i test_mm_ipcvtts_ps_epi8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_ipcvtts_ps_epi8(__A);
}

__m128i test_mm_mask_ipcvtts_ps_epi8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_mask_ipcvtts_ps_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_ps_epi8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_maskz_ipcvtts_ps_epi8(__A, __B);
}

__m256i test_mm256_ipcvtts_ps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_ipcvtts_ps_epi8(__A);
}

__m256i test_mm256_mask_ipcvtts_ps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_mask_ipcvtts_ps_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_ps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_ps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_maskz_ipcvtts_ps_epi8(__A, __B);
}

__m128i test_mm_ipcvtts_ps_epu8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_ipcvtts_ps_epu8(__A);
}

__m128i test_mm_mask_ipcvtts_ps_epu8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_mask_ipcvtts_ps_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtts_ps_epu8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_maskz_ipcvtts_ps_epu8(__A, __B);
}

__m256i test_mm256_ipcvtts_ps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_ipcvtts_ps_epu8(__A);
}

__m256i test_mm256_mask_ipcvtts_ps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_mask_ipcvtts_ps_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtts_ps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtts_ps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_maskz_ipcvtts_ps_epu8(__A, __B);
}

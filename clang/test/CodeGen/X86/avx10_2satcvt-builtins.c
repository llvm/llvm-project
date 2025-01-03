// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-256 \
// RUN: -Wno-invalid-feature-combination -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-256 \
// RUN: -Wno-invalid-feature-combination -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_ipcvtnebf16_epi8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs128
  return _mm_ipcvtnebf16_epi8(__A);
}

__m128i test_mm_mask_ipcvtnebf16_epi8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvtnebf16_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtnebf16_epi8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvtnebf16_epi8(__A, __B);
}

__m256i test_mm256_ipcvtnebf16_epi8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs256
  return _mm256_ipcvtnebf16_epi8(__A);
}

__m256i test_mm256_mask_ipcvtnebf16_epi8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvtnebf16_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtnebf16_epi8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162ibs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvtnebf16_epi8(__A, __B);
}

__m128i test_mm_ipcvtnebf16_epu8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs128
  return _mm_ipcvtnebf16_epu8(__A);
}

__m128i test_mm_mask_ipcvtnebf16_epu8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvtnebf16_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtnebf16_epu8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvtnebf16_epu8(__A, __B);
}

__m256i test_mm256_ipcvtnebf16_epu8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs256
  return _mm256_ipcvtnebf16_epu8(__A);
}

__m256i test_mm256_mask_ipcvtnebf16_epu8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvtnebf16_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtnebf16_epu8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvtnebf162iubs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvtnebf16_epu8(__A, __B);
}

__m128i test_mm_ipcvtph_epi8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_ipcvtph_epi8(__A);
}

__m128i test_mm_mask_ipcvtph_epi8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_mask_ipcvtph_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtph_epi8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs128
  return _mm_maskz_ipcvtph_epi8(__A, __B);
}

__m256i test_mm256_ipcvtph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_ipcvtph_epi8(__A);
}

__m256i test_mm256_mask_ipcvtph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_mask_ipcvtph_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_maskz_ipcvtph_epi8(__A, __B);
}

__m256i test_mm256_ipcvt_roundph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_ipcvt_roundph_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvt_roundph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_mask_ipcvt_roundph_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m256i test_mm256_maskz_ipcvt_roundph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2ibs256
  return _mm256_maskz_ipcvt_roundph_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvtph_epu8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_ipcvtph_epu8(__A);
}

__m128i test_mm_mask_ipcvtph_epu8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_mask_ipcvtph_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtph_epu8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs128
  return _mm_maskz_ipcvtph_epu8(__A, __B);
}

__m256i test_mm256_ipcvtph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_ipcvtph_epu8(__A);
}

__m256i test_mm256_mask_ipcvtph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_mask_ipcvtph_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_maskz_ipcvtph_epu8(__A, __B);
}

__m256i test_mm256_ipcvt_roundph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_ipcvt_roundph_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvt_roundph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_mask_ipcvt_roundph_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m256i test_mm256_maskz_ipcvt_roundph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtph2iubs256
  return _mm256_maskz_ipcvt_roundph_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvtps_epi8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_ipcvtps_epi8(__A);
}

__m128i test_mm_mask_ipcvtps_epi8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_mask_ipcvtps_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtps_epi8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs128
  return _mm_maskz_ipcvtps_epi8(__A, __B);
}

__m256i test_mm256_ipcvtps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_ipcvtps_epi8(__A);
}

__m256i test_mm256_mask_ipcvtps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_mask_ipcvtps_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_maskz_ipcvtps_epi8(__A, __B);
}

__m256i test_mm256_ipcvt_roundps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_ipcvt_roundps_epi8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvt_roundps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_mask_ipcvt_roundps_epi8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvt_roundps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2ibs256
  return _mm256_maskz_ipcvt_roundps_epi8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvtps_epu8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_ipcvtps_epu8(__A);
}

__m128i test_mm_mask_ipcvtps_epu8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_mask_ipcvtps_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvtps_epu8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs128
  return _mm_maskz_ipcvtps_epu8(__A, __B);
}

__m256i test_mm256_ipcvtps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_ipcvtps_epu8(__A);
}

__m256i test_mm256_mask_ipcvtps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_mask_ipcvtps_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvtps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_maskz_ipcvtps_epu8(__A, __B);
}

__m256i test_mm256_ipcvt_roundps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_ipcvt_roundps_epu8(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvt_roundps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_mask_ipcvt_roundps_epu8(__S, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvt_roundps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvtps2iubs256
  return _mm256_maskz_ipcvt_roundps_epu8(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvttnebf16_epi8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs128
  return _mm_ipcvttnebf16_epi8(__A);
}

__m128i test_mm_mask_ipcvttnebf16_epi8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvttnebf16_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttnebf16_epi8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvttnebf16_epi8(__A, __B);
}

__m256i test_mm256_ipcvttnebf16_epi8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs256
  return _mm256_ipcvttnebf16_epi8(__A);
}

__m256i test_mm256_mask_ipcvttnebf16_epi8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvttnebf16_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttnebf16_epi8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttnebf16_epi8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162ibs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvttnebf16_epi8(__A, __B);
}

__m128i test_mm_ipcvttnebf16_epu8(__m128bh __A) {
  // CHECK-LABEL: @test_mm_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs128
  return _mm_ipcvttnebf16_epu8(__A);
}

__m128i test_mm_mask_ipcvttnebf16_epu8(__m128i __S, __mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_ipcvttnebf16_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttnebf16_epu8(__mmask8 __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs128
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_ipcvttnebf16_epu8(__A, __B);
}

__m256i test_mm256_ipcvttnebf16_epu8(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs256
  return _mm256_ipcvttnebf16_epu8(__A);
}

__m256i test_mm256_mask_ipcvttnebf16_epu8(__m256i __S, __mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_ipcvttnebf16_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttnebf16_epu8(__mmask16 __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttnebf16_epu8(
  // CHECK: @llvm.x86.avx10.vcvttnebf162iubs256
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_ipcvttnebf16_epu8(__A, __B);
}

__m128i test_mm_ipcvttph_epi8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_ipcvttph_epi8(__A);
}

__m128i test_mm_mask_ipcvttph_epi8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_mask_ipcvttph_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttph_epi8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs128
  return _mm_maskz_ipcvttph_epi8(__A, __B);
}

__m256i test_mm256_ipcvttph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_ipcvttph_epi8(__A);
}

__m256i test_mm256_mask_ipcvttph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_mask_ipcvttph_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_maskz_ipcvttph_epi8(__A, __B);
}

__m256i test_mm256_ipcvtt_roundph_epi8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_ipcvtt_roundph_epi8(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvtt_roundph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_mask_ipcvtt_roundph_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvtt_roundph_epi8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtt_roundph_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2ibs256
  return _mm256_maskz_ipcvtt_roundph_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvttph_epu8(__m128h __A) {
  // CHECK-LABEL: @test_mm_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_ipcvttph_epu8(__A);
}

__m128i test_mm_mask_ipcvttph_epu8(__m128i __S, __mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_mask_ipcvttph_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttph_epu8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs128
  return _mm_maskz_ipcvttph_epu8(__A, __B);
}

__m256i test_mm256_ipcvttph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_ipcvttph_epu8(__A);
}

__m256i test_mm256_mask_ipcvttph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_mask_ipcvttph_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_maskz_ipcvttph_epu8(__A, __B);
}

__m256i test_mm256_ipcvtt_roundph_epu8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_ipcvtt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_ipcvtt_roundph_epu8(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvtt_roundph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_mask_ipcvtt_roundph_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvtt_roundph_epu8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtt_roundph_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttph2iubs256
  return _mm256_maskz_ipcvtt_roundph_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvttps_epi8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_ipcvttps_epi8(__A);
}

__m128i test_mm_mask_ipcvttps_epi8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_mask_ipcvttps_epi8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttps_epi8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs128
  return _mm_maskz_ipcvttps_epi8(__A, __B);
}

__m256i test_mm256_ipcvttps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_ipcvttps_epi8(__A);
}

__m256i test_mm256_mask_ipcvttps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_mask_ipcvttps_epi8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_maskz_ipcvttps_epi8(__A, __B);
}

__m256i test_mm256_ipcvtt_roundps_epi8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_ipcvtt_roundps_epi8(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvtt_roundps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_mask_ipcvtt_roundps_epi8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvtt_roundps_epi8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtt_roundps_epi8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2ibs256
  return _mm256_maskz_ipcvtt_roundps_epi8(__A, __B, _MM_FROUND_NO_EXC);
}

__m128i test_mm_ipcvttps_epu8(__m128 __A) {
  // CHECK-LABEL: @test_mm_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_ipcvttps_epu8(__A);
}

__m128i test_mm_mask_ipcvttps_epu8(__m128i __S, __mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_mask_ipcvttps_epu8(__S, __A, __B);
}

__m128i test_mm_maskz_ipcvttps_epu8(__mmask8 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs128
  return _mm_maskz_ipcvttps_epu8(__A, __B);
}

__m256i test_mm256_ipcvttps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_ipcvttps_epu8(__A);
}

__m256i test_mm256_mask_ipcvttps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_mask_ipcvttps_epu8(__S, __A, __B);
}

__m256i test_mm256_maskz_ipcvttps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvttps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_maskz_ipcvttps_epu8(__A, __B);
}

__m256i test_mm256_ipcvtt_roundps_epu8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_ipcvtt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_ipcvtt_roundps_epu8(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_ipcvtt_roundps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_ipcvtt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_mask_ipcvtt_roundps_epu8(__S, __A, __B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_ipcvtt_roundps_epu8(__mmask8 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_ipcvtt_roundps_epu8(
  // CHECK: @llvm.x86.avx10.mask.vcvttps2iubs256
  return _mm256_maskz_ipcvtt_roundps_epu8(__A, __B, _MM_FROUND_NO_EXC);
}

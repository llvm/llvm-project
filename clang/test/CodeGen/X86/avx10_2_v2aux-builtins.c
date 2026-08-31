// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10v2aux \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10v2aux \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_cvtps_bf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8128(<4 x float> %{{.*}})
  return _mm_cvtps_bf8(__A);
}

__m128i test_mm_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtps_bf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtps_bf8(__U, __A);
}

__m128i test_mm256_cvtps_bf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8256(<8 x float> %{{.*}})
  return _mm256_cvtps_bf8(__A);
}

__m128i test_mm256_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtps_bf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtps_bf8(__U, __A);
}

__m128i test_mm512_cvtps_bf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8512(<16 x float> %{{.*}})
  return _mm512_cvtps_bf8(__A);
}

__m128i test_mm512_mask_cvtps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2bf8512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtps_bf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2bf8512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvtps_bf8(__U, __A);
}

__m128i test_mm_cvts_ps_bf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8s128(<4 x float> %{{.*}})
  return _mm_cvts_ps_bf8(__A);
}

__m128i test_mm_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_ps_bf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_ps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvts_ps_bf8(__U, __A);
}

__m128i test_mm256_cvts_ps_bf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8s256(<8 x float> %{{.*}})
  return _mm256_cvts_ps_bf8(__A);
}

__m128i test_mm256_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_ps_bf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_ps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvts_ps_bf8(__U, __A);
}

__m128i test_mm512_cvts_ps_bf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2bf8s512(<16 x float> %{{.*}})
  return _mm512_cvts_ps_bf8(__A);
}

__m128i test_mm512_mask_cvts_ps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_ps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2bf8s512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_ps_bf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_ps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2bf8s512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvts_ps_bf8(__U, __A);
}

__m128i test_mm_cvtps_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8128(<4 x float> %{{.*}})
  return _mm_cvtps_hf8(__A);
}

__m128i test_mm_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtps_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtps_hf8(__U, __A);
}

__m128i test_mm256_cvtps_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8256(<8 x float> %{{.*}})
  return _mm256_cvtps_hf8(__A);
}

__m128i test_mm256_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtps_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtps_hf8(__U, __A);
}

__m128i test_mm512_cvtps_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8512(<16 x float> %{{.*}})
  return _mm512_cvtps_hf8(__A);
}

__m128i test_mm512_mask_cvtps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2hf8512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtps_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2hf8512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvtps_hf8(__U, __A);
}

__m128i test_mm_cvts_ps_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8s128(<4 x float> %{{.*}})
  return _mm_cvts_ps_hf8(__A);
}

__m128i test_mm_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_ps_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_ps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvts_ps_hf8(__U, __A);
}

__m128i test_mm256_cvts_ps_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8s256(<8 x float> %{{.*}})
  return _mm256_cvts_ps_hf8(__A);
}

__m128i test_mm256_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_ps_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_ps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvts_ps_hf8(__U, __A);
}

__m128i test_mm512_cvts_ps_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtps2hf8s512(<16 x float> %{{.*}})
  return _mm512_cvts_ps_hf8(__A);
}

__m128i test_mm512_mask_cvts_ps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_ps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2hf8s512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_ps_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_ps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtps2hf8s512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvts_ps_hf8(__U, __A);
}

__m128i test_mm_cvtrops_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8128(<4 x float> %{{.*}})
  return _mm_cvtrops_hf8(__A);
}

__m128i test_mm_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtrops_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtrops_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtrops_hf8(__U, __A);
}

__m128i test_mm256_cvtrops_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8256(<8 x float> %{{.*}})
  return _mm256_cvtrops_hf8(__A);
}

__m128i test_mm256_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtrops_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtrops_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtrops_hf8(__U, __A);
}

__m128i test_mm512_cvtrops_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8512(<16 x float> %{{.*}})
  return _mm512_cvtrops_hf8(__A);
}

__m128i test_mm512_mask_cvtrops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtrops_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtrops_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtrops_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvtrops_hf8(__U, __A);
}

__m128i test_mm_cvts_rops_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8s128(<4 x float> %{{.*}})
  return _mm_cvts_rops_hf8(__A);
}

__m128i test_mm_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_rops_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_rops_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s128(<4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvts_rops_hf8(__U, __A);
}

__m128i test_mm256_cvts_rops_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8s256(<8 x float> %{{.*}})
  return _mm256_cvts_rops_hf8(__A);
}

__m128i test_mm256_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_rops_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_rops_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s256(<8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvts_rops_hf8(__U, __A);
}

__m128i test_mm512_cvts_rops_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8s512(<16 x float> %{{.*}})
  return _mm512_cvts_rops_hf8(__A);
}

__m128i test_mm512_mask_cvts_rops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_rops_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8s512(<16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_rops_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_rops_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtrops2hf8s512(<16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvts_rops_hf8(__U, __A);
}

__m128i test_mm_cvtbiasps_bf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}})
  return _mm_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtbiasps_bf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasps_bf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtbiasps_bf8(__U, __A, __B);
}

__m128i test_mm512_cvtbiasps_bf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm512_mask_cvtbiasps_bf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvtbiasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvtbiasps_bf8(__U, __A, __B);
}

__m128i test_mm_cvts_biasps_bf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}})
  return _mm_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts_biasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts_biasps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvts_biasps_bf8(__U, __A, __B);
}

__m128i test_mm256_cvts_biasps_bf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm256_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvts_biasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_biasps_bf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvts_biasps_bf8(__U, __A, __B);
}

__m128i test_mm512_cvts_biasps_bf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm512_mask_cvts_biasps_bf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvts_biasps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvts_biasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_biasps_bf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2bf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvts_biasps_bf8(__U, __A, __B);
}

__m128i test_mm_cvtbiasps_hf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}})
  return _mm_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtbiasps_hf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasps_hf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtbiasps_hf8(__U, __A, __B);
}

__m128i test_mm512_cvtbiasps_hf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm512_mask_cvtbiasps_hf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvtbiasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvtbiasps_hf8(__U, __A, __B);
}

__m128i test_mm_cvts_biasps_hf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}})
  return _mm_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts_biasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts_biasps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s128(<4 x i32> %{{.*}}, <4 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvts_biasps_hf8(__U, __A, __B);
}

__m128i test_mm256_cvts_biasps_hf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm256_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvts_biasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_biasps_hf8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s256(<8 x i32> %{{.*}}, <8 x float> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvts_biasps_hf8(__U, __A, __B);
}

__m128i test_mm512_cvts_biasps_hf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm512_mask_cvts_biasps_hf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvts_biasps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvts_biasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_biasps_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbiasps2hf8s512(<16 x i32> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm512_maskz_cvts_biasps_hf8(__U, __A, __B);
}

__m128 test_mm_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.vcvtbf82ps128(<16 x i8> %{{.*}})
  return _mm_cvtbf8_ps(__A);
}

__m128 test_mm_mask_cvtbf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <4 x float> @llvm.x86.avx10.vcvtbf82ps128(<16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> [[RES]], <4 x float> %{{.*}}
  return _mm_mask_cvtbf8_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <4 x float> @llvm.x86.avx10.vcvtbf82ps128(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> [[RES]], <4 x float> %{{.*}}
  return _mm_maskz_cvtbf8_ps(__U, __A);
}

__m256 test_mm256_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.vcvtbf82ps256(<16 x i8> %{{.*}})
  return _mm256_cvtbf8_ps(__A);
}

__m256 test_mm256_mask_cvtbf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <8 x float> @llvm.x86.avx10.vcvtbf82ps256(<16 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> [[RES]], <8 x float> %{{.*}}
  return _mm256_mask_cvtbf8_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <8 x float> @llvm.x86.avx10.vcvtbf82ps256(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> [[RES]], <8 x float> %{{.*}}
  return _mm256_maskz_cvtbf8_ps(__U, __A);
}

__m512 test_mm512_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.vcvtbf82ps512(<16 x i8> %{{.*}})
  return _mm512_cvtbf8_ps(__A);
}

__m512 test_mm512_mask_cvtbf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <16 x float> @llvm.x86.avx10.vcvtbf82ps512(<16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> [[RES]], <16 x float> %{{.*}}
  return _mm512_mask_cvtbf8_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_cvtbf8_ps(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf8_ps(
  // CHECK: [[RES:%.*]] = call <16 x float> @llvm.x86.avx10.vcvtbf82ps512(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> [[RES]], <16 x float> %{{.*}}
  return _mm512_maskz_cvtbf8_ps(__U, __A);
}

__m128 test_mm_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.vcvthf82ps128(<16 x i8> %{{.*}})
  return _mm_cvthf8_ps(__A);
}

__m128 test_mm_mask_cvthf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <4 x float> @llvm.x86.avx10.vcvthf82ps128(<16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> [[RES]], <4 x float> %{{.*}}
  return _mm_mask_cvthf8_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <4 x float> @llvm.x86.avx10.vcvthf82ps128(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> [[RES]], <4 x float> %{{.*}}
  return _mm_maskz_cvthf8_ps(__U, __A);
}

__m256 test_mm256_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvthf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.vcvthf82ps256(<16 x i8> %{{.*}})
  return _mm256_cvthf8_ps(__A);
}

__m256 test_mm256_mask_cvthf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <8 x float> @llvm.x86.avx10.vcvthf82ps256(<16 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> [[RES]], <8 x float> %{{.*}}
  return _mm256_mask_cvthf8_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <8 x float> @llvm.x86.avx10.vcvthf82ps256(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> [[RES]], <8 x float> %{{.*}}
  return _mm256_maskz_cvthf8_ps(__U, __A);
}

__m512 test_mm512_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvthf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.vcvthf82ps512(<16 x i8> %{{.*}})
  return _mm512_cvthf8_ps(__A);
}

__m512 test_mm512_mask_cvthf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <16 x float> @llvm.x86.avx10.vcvthf82ps512(<16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> [[RES]], <16 x float> %{{.*}}
  return _mm512_mask_cvthf8_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_cvthf8_ps(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvthf8_ps(
  // CHECK: [[RES:%.*]] = call <16 x float> @llvm.x86.avx10.vcvthf82ps512(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> [[RES]], <16 x float> %{{.*}}
  return _mm512_maskz_cvthf8_ps(__U, __A);
}

__m128i test_mm_cvts_bf8_bf4(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvts_bf8_bf4(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf82bf4s128(<16 x i8> %{{.*}})
  return _mm_cvts_bf8_bf4(__A);
}

__m128i test_mm256_cvts_bf8_bf4(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvts_bf8_bf4(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf82bf4s256(<32 x i8> %{{.*}})
  return _mm256_cvts_bf8_bf4(__A);
}

__m256i test_mm512_cvts_bf8_bf4(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvts_bf8_bf4(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf82bf4s512(<64 x i8> %{{.*}})
  return _mm512_cvts_bf8_bf4(__A);
}

__m128i test_mm_cvts_hf8_bf4(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvts_hf8_bf4(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf82bf4s128(<16 x i8> %{{.*}})
  return _mm_cvts_hf8_bf4(__A);
}

__m128i test_mm256_cvts_hf8_bf4(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvts_hf8_bf4(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf82bf4s256(<32 x i8> %{{.*}})
  return _mm256_cvts_hf8_bf4(__A);
}

__m256i test_mm512_cvts_hf8_bf4(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvts_hf8_bf4(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf82bf4s512(<64 x i8> %{{.*}})
  return _mm512_cvts_hf8_bf4(__A);
}

__m128i test_mm_cvts_bf8_bf6(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvts_bf8_bf6(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf82bf6s128(<16 x i8> %{{.*}})
  return _mm_cvts_bf8_bf6(__A);
}

__m256i test_mm256_cvts_bf8_bf6(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvts_bf8_bf6(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf82bf6s256(<32 x i8> %{{.*}})
  return _mm256_cvts_bf8_bf6(__A);
}

__m512i test_mm512_cvts_bf8_bf6(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvts_bf8_bf6(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf82bf6s512(<64 x i8> %{{.*}})
  return _mm512_cvts_bf8_bf6(__A);
}

__m128i test_mm_cvts_hf8_hf6(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvts_hf8_hf6(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf82hf6s128(<16 x i8> %{{.*}})
  return _mm_cvts_hf8_hf6(__A);
}

__m256i test_mm256_cvts_hf8_hf6(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvts_hf8_hf6(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf82hf6s256(<32 x i8> %{{.*}})
  return _mm256_cvts_hf8_hf6(__A);
}

__m512i test_mm512_cvts_hf8_hf6(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvts_hf8_hf6(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf82hf6s512(<64 x i8> %{{.*}})
  return _mm512_cvts_hf8_hf6(__A);
}

__m128i test_mm_cvtbf4_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf4_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8128(<16 x i8> %{{.*}})
  return _mm_cvtbf4_hf8(__A);
}

__m128i test_mm_mask_cvtbf4_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8128(<16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_mask_cvtbf4_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtbf4_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8128(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_maskz_cvtbf4_hf8(__U, __A);
}

__m256i test_mm256_cvtbf4_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf4_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8256(<16 x i8> %{{.*}})
  return _mm256_cvtbf4_hf8(__A);
}

__m256i test_mm256_mask_cvtbf4_hf8(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8256(<16 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_mask_cvtbf4_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvtbf4_hf8(__mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8256(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_maskz_cvtbf4_hf8(__U, __A);
}

__m512i test_mm512_cvtbf4_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf4_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8512(<32 x i8> %{{.*}})
  return _mm512_cvtbf4_hf8(__A);
}

__m512i test_mm512_mask_cvtbf4_hf8(__m512i __W, __mmask64 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8512(<32 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_mask_cvtbf4_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvtbf4_hf8(__mmask64 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf4_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8512(<32 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_maskz_cvtbf4_hf8(__U, __A);
}

__m128i test_mm_cvtbf6_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8128(<16 x i8> %{{.*}})
  return _mm_cvtbf6_hf8(__A);
}

__m128i test_mm_mask_cvtbf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8128(<16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_mask_cvtbf6_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtbf6_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8128(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_maskz_cvtbf6_hf8(__U, __A);
}

__m256i test_mm256_cvtbf6_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8256(<32 x i8> %{{.*}})
  return _mm256_cvtbf6_hf8(__A);
}

__m256i test_mm256_mask_cvtbf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8256(<32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_mask_cvtbf6_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvtbf6_hf8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8256(<32 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_maskz_cvtbf6_hf8(__U, __A);
}

__m512i test_mm512_cvtbf6_hf8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8512(<64 x i8> %{{.*}})
  return _mm512_cvtbf6_hf8(__A);
}

__m512i test_mm512_mask_cvtbf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8512(<64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_mask_cvtbf6_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvtbf6_hf8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf6_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8512(<64 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_maskz_cvtbf6_hf8(__U, __A);
}

__m128i test_mm_cvthf6_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf62hf8128(<16 x i8> %{{.*}})
  return _mm_cvthf6_hf8(__A);
}

__m128i test_mm_mask_cvthf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvthf62hf8128(<16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_mask_cvthf6_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvthf6_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.x86.avx10.vcvthf62hf8128(<16 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[RES]], <16 x i8> %{{.*}}
  return _mm_maskz_cvthf6_hf8(__U, __A);
}

__m256i test_mm256_cvthf6_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvthf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf62hf8256(<32 x i8> %{{.*}})
  return _mm256_cvthf6_hf8(__A);
}

__m256i test_mm256_mask_cvthf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvthf62hf8256(<32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_mask_cvthf6_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvthf6_hf8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.x86.avx10.vcvthf62hf8256(<32 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[RES]], <32 x i8> %{{.*}}
  return _mm256_maskz_cvthf6_hf8(__U, __A);
}

__m512i test_mm512_cvthf6_hf8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvthf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf62hf8512(<64 x i8> %{{.*}})
  return _mm512_cvthf6_hf8(__A);
}

__m512i test_mm512_mask_cvthf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvthf62hf8512(<64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_mask_cvthf6_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvthf6_hf8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvthf6_hf8(
  // CHECK: [[RES:%.*]] = call <64 x i8> @llvm.x86.avx10.vcvthf62hf8512(<64 x i8> %{{.*}})
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[RES]], <64 x i8> %{{.*}}
  return _mm512_maskz_cvthf6_hf8(__U, __A);
}

__m128i test_mm_unpack_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_unpack_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(<16 x i8> %{{.*}}, i8 1)
  return _mm_unpack_epi8(__A, 1);
}

__m128i test_mm_mask_unpack_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_unpack_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(<16 x i8> %{{.*}}, i8 1)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpack_epi8(__W, __U, __A, 1);
}

__m128i test_mm_maskz_unpack_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_unpack_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(<16 x i8> %{{.*}}, i8 1)
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpack_epi8(__U, __A, 1);
}

__m256i test_mm256_unpack_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_unpack_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(<32 x i8> %{{.*}}, i8 2)
  return _mm256_unpack_epi8(__A, 2);
}

__m256i test_mm256_mask_unpack_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_unpack_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(<32 x i8> %{{.*}}, i8 2)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpack_epi8(__W, __U, __A, 2);
}

__m256i test_mm256_maskz_unpack_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_unpack_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(<32 x i8> %{{.*}}, i8 2)
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpack_epi8(__U, __A, 2);
}

__m512i test_mm512_unpack_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_unpack_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(<64 x i8> %{{.*}}, i8 3)
  return _mm512_unpack_epi8(__A, 3);
}

__m512i test_mm512_mask_unpack_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_unpack_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(<64 x i8> %{{.*}}, i8 3)
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_unpack_epi8(__W, __U, __A, 3);
}

__m512i test_mm512_maskz_unpack_epi8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_unpack_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(<64 x i8> %{{.*}}, i8 3)
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_unpack_epi8(__U, __A, 3);
}

__m512i test_mm512_unpack_epi8_compose(__m512i __A) {
  // CHECK-LABEL: @test_mm512_unpack_epi8_compose(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(<64 x i8> %{{.*}}, i8 45)
  return _mm512_unpack_epi8(
      __A, _MM_UNPACKB_SIZE(3) | _MM_UNPACKB_START(1) | _MM_UNPACKB_SEXT);
}

__m256i test_mm256_unpack_epi8_compose(__m256i __A) {
  // CHECK-LABEL: @test_mm256_unpack_epi8_compose(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(<32 x i8> %{{.*}}, i8 9)
  return _mm256_unpack_epi8(__A, _MM_UNPACKB_SIZE(2) | _MM_UNPACKB_START(1));
}

__m128i test_mm_unpack_epi8_compose(__m128i __A) {
  // CHECK-LABEL: @test_mm_unpack_epi8_compose(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(<16 x i8> %{{.*}}, i8 60)
  return _mm_unpack_epi8(__A, _MM_UNPACKB_SIZE(7) | _MM_UNPACKB_SEXT);
}

__m128i test_mm_cvtss_epi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 -1)
  return _mm_cvtss_epi32_epi8(__A);
}

__m128i test_mm_mask_cvtss_epi32_epi8(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_mask_cvtss_epi32_epi8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtss_epi32_epi8(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtss_epi32_epi8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm_maskz_cvtss_epi32_epi8(__U, __A);
}

__m128i test_mm256_cvtss_epi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.256(<8 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 -1)
  return _mm256_cvtss_epi32_epi8(__A);
}

__m128i test_mm256_mask_cvtss_epi32_epi8(__m128i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.256(<8 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_mask_cvtss_epi32_epi8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtss_epi32_epi8(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtss_epi32_epi8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.256(<8 x i32> %{{.*}}, <16 x i8> %{{.*}}, i8 %{{.*}})
  return _mm256_maskz_cvtss_epi32_epi8(__U, __A);
}

__m128i test_mm512_cvtss_epi32_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.512(<16 x i32> %{{.*}}, <16 x i8> %{{.*}}, i16 -1)
  return _mm512_cvtss_epi32_epi8(__A);
}

__m128i test_mm512_mask_cvtss_epi32_epi8(__m128i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtss_epi32_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.512(<16 x i32> %{{.*}}, <16 x i8> %{{.*}}, i16 %{{.*}})
  return _mm512_mask_cvtss_epi32_epi8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtss_epi32_epi8(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtss_epi32_epi8(
  // CHECK: zeroinitializer
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.pmovss.db.512(<16 x i32> %{{.*}}, <16 x i8> %{{.*}}, i16 %{{.*}})
  return _mm512_maskz_cvtss_epi32_epi8(__U, __A);
}

void test_mm_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtss_epi32_storeu_epi8(
  // CHECK: call void @llvm.x86.avx10.mask.pmovss.db.mem.128(ptr %{{.*}}, <4 x i32> %{{.*}}, i8 %{{.*}})
  _mm_mask_cvtss_epi32_storeu_epi8(__P, __M, __A);
}

void test_mm256_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtss_epi32_storeu_epi8(
  // CHECK: call void @llvm.x86.avx10.mask.pmovss.db.mem.256(ptr %{{.*}}, <8 x i32> %{{.*}}, i8 %{{.*}})
  _mm256_mask_cvtss_epi32_storeu_epi8(__P, __M, __A);
}

void test_mm512_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtss_epi32_storeu_epi8(
  // CHECK: call void @llvm.x86.avx10.mask.pmovss.db.mem.512(ptr %{{.*}}, <16 x i32> %{{.*}}, i16 %{{.*}})
  _mm512_mask_cvtss_epi32_storeu_epi8(__P, __M, __A);
}

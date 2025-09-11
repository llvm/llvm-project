// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s  --check-prefixes=CHECK

#include <immintrin.h>
#include <stddef.h>

__m128i test_mm_cvtts_pd_epi32(__m128d A){
// CHECK-LABEL: @test_mm_cvtts_pd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
  return _mm_cvtts_pd_epi32(A);
}

__m128i test_mm_mask_cvtts_pd_epi32(__m128i W, __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_mask_cvtts_pd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
    return _mm_mask_cvtts_pd_epi32(W,U,A);
}

__m128i test_mm_maskz_cvtts_pd_epi32( __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_maskz_cvtts_pd_epi32(
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
    return _mm_maskz_cvtts_pd_epi32(U,A);
}

__m128i test_mm256_cvtts_pd_epi32(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_pd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.256(<4 x double>
  return _mm256_cvtts_pd_epi32(A);
}

__m128i test_mm256_mask_cvtts_pd_epi32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_pd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.256(<4 x double>
    return _mm256_mask_cvtts_pd_epi32(W,U,A);
}

__m128i test_mm256_maskz_cvtts_pd_epi32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_pd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.256(<4 x double>
    return _mm256_maskz_cvtts_pd_epi32(U,A);
}

__m128i test_mm_cvtts_pd_epu32(__m128d A){
// CHECK-LABEL: @test_mm_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
  return _mm_cvtts_pd_epu32(A);
}

__m128i test_mm_mask_cvtts_pd_epu32(__m128i W, __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_mask_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
    return _mm_mask_cvtts_pd_epu32(W,U,A);
}

__m128i test_mm_maskz_cvtts_pd_epu32( __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_maskz_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
    return _mm_maskz_cvtts_pd_epu32(U,A);
}


__m128i test_mm256_cvtts_pd_epu32(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.256(<4 x double>
  return _mm256_cvtts_pd_epu32(A);
}

__m128i test_mm256_mask_cvtts_pd_epu32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.256(<4 x double>
    return _mm256_mask_cvtts_pd_epu32(W,U,A);
}

__m128i test_mm256_maskz_cvtts_pd_epu32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_pd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.256(<4 x double>
    return _mm256_maskz_cvtts_pd_epu32(U,A);
}

__m128i test_mm_cvtts_ps_epi32(__m128 A){
// CHECK-LABEL: @test_mm_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
  return _mm_cvtts_ps_epi32(A);
}

__m128i test_mm_mask_cvtts_ps_epi32(__m128i W, __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_mask_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
    return _mm_mask_cvtts_ps_epi32(W,U,A);
}

__m128i test_mm_maskz_cvtts_ps_epi32( __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_maskz_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
    return _mm_maskz_cvtts_ps_epi32(U,A);
}

__m256i test_mm256_cvtts_ps_epi32(__m256 A){
// CHECK-LABEL: @test_mm256_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.256(<8 x float>
  return _mm256_cvtts_ps_epi32(A);
}

__m256i test_mm256_mask_cvtts_ps_epi32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.256(<8 x float>
    return _mm256_mask_cvtts_ps_epi32(W,U,A);
}

__m256i test_mm256_maskz_cvtts_ps_epi32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_ps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.256(<8 x float>
    return _mm256_maskz_cvtts_ps_epi32(U,A);
}

__m128i test_mm_cvtts_ps_epu32(__m128 A){
// CHECK-LABEL: @test_mm_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
  return _mm_cvtts_ps_epu32(A);
}

__m128i test_mm_mask_cvtts_ps_epu32(__m128i W, __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_mask_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
    return _mm_mask_cvtts_ps_epu32(W,U,A);
}

__m128i test_mm_maskz_cvtts_ps_epu32( __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_maskz_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
    return _mm_maskz_cvtts_ps_epu32(U,A);
}

__m256i test_mm256_cvtts_ps_epu32(__m256 A){
// CHECK-LABEL: @test_mm256_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.256(<8 x float>
  return _mm256_cvtts_ps_epu32(A);
}

__m256i test_mm256_mask_cvtts_ps_epu32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.256(<8 x float>
    return _mm256_mask_cvtts_ps_epu32(W,U,A);
}

__m256i test_mm256_maskz_cvtts_ps_epu32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_ps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.256(<8 x float>
    return _mm256_maskz_cvtts_ps_epu32(U,A);
}

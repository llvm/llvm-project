// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386 -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s  --check-prefixes=CHECK,X64

#include <immintrin.h>
#include <stddef.h>

__m128i test_mm_cvttspd_epi32(__m128d A){
// CHECK-LABEL: @test_mm_cvttspd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
  return _mm_cvttspd_epi32(A);
}

__m128i test_mm_mask_cvttspd_epi32(__m128i W, __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_mask_cvttspd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
    return _mm_mask_cvttspd_epi32(W,U,A);
}

__m128i test_mm_maskz_cvttspd_epi32( __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_maskz_cvttspd_epi32(
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.128(<2 x double>
    return _mm_maskz_cvttspd_epi32(U,A);
}

__m128i test_mm256_cvttspd_epi32(__m256d A){
// CHECK-LABEL: @test_mm256_cvttspd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
  return _mm256_cvttspd_epi32(A);
}

__m128i test_mm256_mask_cvttspd_epi32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvttspd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
    return _mm256_mask_cvttspd_epi32(W,U,A);
}

__m128i test_mm256_maskz_cvttspd_epi32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvttspd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
    return _mm256_maskz_cvttspd_epi32(U,A);
}

__m128i test_mm256_cvtts_roundpd_epi32(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
    return _mm256_cvtts_roundpd_epi32(A, _MM_FROUND_NEARBYINT);
}

__m128i test_mm256_mask_cvtts_roundpd_epi32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
    return _mm256_mask_cvtts_roundpd_epi32(W,U,A,_MM_FROUND_NEARBYINT);
}

__m128i test_mm256_maskz_cvtts_roundpd_epi32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dqs.round.256(<4 x double>
    return _mm256_maskz_cvtts_roundpd_epi32(U,A,_MM_FROUND_NEARBYINT);
}

__m128i test_mm_cvttspd_epu32(__m128d A){
// CHECK-LABEL: @test_mm_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
  return _mm_cvttspd_epu32(A);
}

__m128i test_mm_mask_cvttspd_epu32(__m128i W, __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_mask_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
    return _mm_mask_cvttspd_epu32(W,U,A);
}

__m128i test_mm_maskz_cvttspd_epu32( __mmask8 U, __m128d A){
// CHECK-LABEL: @test_mm_maskz_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.128(<2 x double>
    return _mm_maskz_cvttspd_epu32(U,A);
}


__m128i test_mm256_cvttspd_epu32(__m256d A){
// CHECK-LABEL: @test_mm256_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
  return _mm256_cvttspd_epu32(A);
}

__m128i test_mm256_mask_cvttspd_epu32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
    return _mm256_mask_cvttspd_epu32(W,U,A);
}

__m128i test_mm256_maskz_cvttspd_epu32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvttspd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
    return _mm256_maskz_cvttspd_epu32(U,A);
}

__m128i test_mm256_cvtts_roundpd_epu32(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
    return _mm256_cvtts_roundpd_epu32(A, _MM_FROUND_NEARBYINT);
}

__m128i test_mm256_mask_cvtts_roundpd_epu32(__m128i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
    return _mm256_mask_cvtts_roundpd_epu32(W,U,A,_MM_FROUND_NEARBYINT);
}

__m128i test_mm256_maskz_cvtts_roundpd_epu32(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udqs.round.256(<4 x double>
    return _mm256_maskz_cvtts_roundpd_epu32(U,A,_MM_FROUND_NEARBYINT);
}

__m128i test_mm_cvttsps_epi32(__m128 A){
// CHECK-LABEL: @test_mm_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
  return _mm_cvttsps_epi32(A);
}

__m128i test_mm_mask_cvttsps_epi32(__m128i W, __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_mask_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
    return _mm_mask_cvttsps_epi32(W,U,A);
}

__m128i test_mm_maskz_cvttsps_epi32( __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_maskz_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.128(<4 x float>
    return _mm_maskz_cvttsps_epi32(U,A);
}

__m256i test_mm256_cvttsps_epi32(__m256 A){
// CHECK-LABEL: @test_mm256_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
  return _mm256_cvttsps_epi32(A);
}

__m256i test_mm256_mask_cvttsps_epi32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
    return _mm256_mask_cvttsps_epi32(W,U,A);
}

__m256i test_mm256_maskz_cvttsps_epi32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvttsps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
    return _mm256_maskz_cvttsps_epi32(U,A);
}

__m256i test_mm256_cvtts_roundps_epi32(__m256 A){
// CHECK-LABEL: @test_mm256_cvtts_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
    return _mm256_cvtts_roundps_epi32(A, _MM_FROUND_NEARBYINT);
}

__m256i test_mm256_mask_cvtts_roundps_epi32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
    return _mm256_mask_cvtts_roundps_epi32(W,U,A,_MM_FROUND_NEARBYINT);
}

__m256i test_mm256_maskz_cvtts_roundps_epi32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dqs.round.256(<8 x float>
    return _mm256_maskz_cvtts_roundps_epi32(U,A,_MM_FROUND_NEARBYINT);
}

__m128i test_mm_cvttsps_epu32(__m128 A){
// CHECK-LABEL: @test_mm_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
  return _mm_cvttsps_epu32(A);
}

__m128i test_mm_mask_cvttsps_epu32(__m128i W, __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_mask_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
    return _mm_mask_cvttsps_epu32(W,U,A);
}

__m128i test_mm_maskz_cvttsps_epu32( __mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm_maskz_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.128(<4 x float>
    return _mm_maskz_cvttsps_epu32(U,A);
}

__m256i test_mm256_cvttsps_epu32(__m256 A){
// CHECK-LABEL: @test_mm256_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
  return _mm256_cvttsps_epu32(A);
}

__m256i test_mm256_mask_cvttsps_epu32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
    return _mm256_mask_cvttsps_epu32(W,U,A);
}

__m256i test_mm256_maskz_cvttsps_epu32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvttsps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
    return _mm256_maskz_cvttsps_epu32(U,A);
}

__m256i test_mm256_cvtts_roundps_epu32(__m256 A){
// CHECK-LABEL: @test_mm256_cvtts_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
    return _mm256_cvtts_roundps_epu32(A, _MM_FROUND_NEARBYINT);
}

__m256i test_mm256_mask_cvtts_roundps_epu32(__m256i W,__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
    return _mm256_mask_cvtts_roundps_epu32(W,U,A,_MM_FROUND_NEARBYINT);
}

__m256i test_mm256_maskz_cvtts_roundps_epu32(__mmask8 U, __m256 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udqs.round.256(<8 x float>
    return _mm256_maskz_cvtts_roundps_epu32(U,A,_MM_FROUND_NEARBYINT);
}

// X64: {{.*}}
// X86: {{.*}}

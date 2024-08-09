// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

// 128 bit
__m128i test_mm_cvttspd_epi64(__m128d A){
    // CHECK-LABEL: @test_mm_cvttspd_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.128(<2 x double>
    return _mm_cvttspd_epi64(A);
}

__m128i test_mm_mask_cvttspd_epi64(__m128i W, __mmask8 U, __m128d A){
    // CHECK-LABEL: @test_mm_mask_cvttspd_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.128(<2 x double>
    return _mm_mask_cvttspd_epi64(W, U,  A);
}

__m128i test_mm_maskz_cvttspd_epi64(__mmask8 U,__m128d A){
    // CHECK-LABEL: @test_mm_maskz_cvttspd_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.128(<2 x double>
    return _mm_maskz_cvttspd_epi64(U, A);
}

__m128i test_mm_cvttspd_epu64(__m128d A){
    // CHECK-LABEL: @test_mm_cvttspd_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.128(<2 x double>
    return _mm_cvttspd_epu64( A);
}

__m128i test_mm_mask_cvttspd_epu64(__m128i W, __mmask8 U, __m128d A){
    // CHECK-LABEL: @test_mm_mask_cvttspd_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.128(<2 x double>
    return _mm_mask_cvttspd_epu64(W, U,  A);
}

__m128i test_mm_maskz_cvttspd_epu64(__mmask8 U,__m128d A){
    // CHECK-LABEL: @test_mm_maskz_cvttspd_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.128(<2 x double>
    return _mm_maskz_cvttspd_epu64(U, A);
}

// 256 bit
__m256i test_mm256_cvttspd_epi64(__m256d A){
// CHECK-LABEL: @test_mm256_cvttspd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_cvttspd_epi64( A);
}

__m256i test_mm256_mask_cvttspd_epi64(__m256i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvttspd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_mask_cvttspd_epi64(W,U, A);
}

__m256i test_mm256_maskz_cvttspd_epi64(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvttspd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_maskz_cvttspd_epi64(U, A);
}

__m256i test_mm256_cvtts_roundpd_epi64(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_roundpd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_cvtts_roundpd_epi64(A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_mask_cvtts_roundpd_epi64(__m256i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundpd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_mask_cvtts_roundpd_epi64(W,U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_maskz_cvtts_roundpd_epi64(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundpd_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2qqs.round.256(<4 x double>
    return _mm256_maskz_cvtts_roundpd_epi64(U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_cvttspd_epu64(__m256d A){
// CHECK-LABEL: @test_mm256_cvttspd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_cvttspd_epu64( A);
}

__m256i test_mm256_mask_cvttspd_epu64(__m256i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvttspd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_mask_cvttspd_epu64(W,U, A);
}

__m256i test_mm256_maskz_cvttspd_epu64(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvttspd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_maskz_cvttspd_epu64(U, A);
}

__m256i test_mm256_cvtts_roundpd_epu64(__m256d A){
// CHECK-LABEL: @test_mm256_cvtts_roundpd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_cvtts_roundpd_epu64(A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_mask_cvtts_roundpd_epu64(__m256i W,__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundpd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_mask_cvtts_roundpd_epu64(W,U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_maskz_cvtts_roundpd_epu64(__mmask8 U, __m256d A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundpd_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttpd2uqqs.round.256(<4 x double>
    return _mm256_maskz_cvtts_roundpd_epu64(U,A,_MM_FROUND_NEARBYINT );
}

// 128 bit
__m128i test_mm_cvttsps_epi64(__m128 A){
    // CHECK-LABEL: @test_mm_cvttsps_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.128(<4 x float>
    return _mm_cvttsps_epi64( A);
}

__m128i test_mm_mask_cvttsps_epi64(__m128i W, __mmask8 U, __m128 A){
    // CHECK-LABEL: @test_mm_mask_cvttsps_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.128(<4 x float>
    return _mm_mask_cvttsps_epi64(W, U,  A);
}

__m128i test_mm_maskz_cvttsps_epi64(__mmask8 U,__m128 A){
    // CHECK-LABEL: @test_mm_maskz_cvttsps_epi64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.128(<4 x float>
    return _mm_maskz_cvttsps_epi64(U, A);
}

__m128i test_mm_cvttsps_epu64(__m128 A){
    // CHECK-LABEL: @test_mm_cvttsps_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.128(<4 x float>
    return _mm_cvttsps_epu64( A);
}

__m128i test_mm_mask_cvttsps_epu64(__m128i W, __mmask8 U, __m128 A){
    // CHECK-LABEL: @test_mm_mask_cvttsps_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.128(<4 x float>
    return _mm_mask_cvttsps_epu64(W, U,  A);
}

__m128i test_mm_maskz_cvttsps_epu64(__mmask8 U,__m128 A){
    // CHECK-LABEL: @test_mm_maskz_cvttsps_epu64
    // CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.128(<4 x float>
    return _mm_maskz_cvttsps_epu64(U, A);
}

__m256i test_mm256_cvttsps_epi64(__m128 A){
// CHECK-LABEL: @test_mm256_cvttsps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
  return _mm256_cvttsps_epi64( A);
}

__m256i test_mm256_mask_cvttsps_epi64(__m256i W,__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_mask_cvttsps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
    return _mm256_mask_cvttsps_epi64(W,U, A);
}

__m256i test_mm256_maskz_cvttsps_epi64(__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_maskz_cvttsps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
    return _mm256_maskz_cvttsps_epi64(U, A);
}

__m256i test_mm256_cvtts_roundps_epi64(__m128 A){
// CHECK-LABEL: @test_mm256_cvtts_roundps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
    return _mm256_cvtts_roundps_epi64(A, _MM_FROUND_NEARBYINT );
}

__m256i test_mm256_mask_cvtts_roundps_epi64(__m256i W,__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
    return _mm256_mask_cvtts_roundps_epi64(W,U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_maskz_cvtts_roundps_epi64(__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundps_epi64
// CHECK: @llvm.x86.avx512.mask.vcvttps2qqs.round.256(<4 x float>
    return _mm256_maskz_cvtts_roundps_epi64(U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_cvttsps_epu64(__m128 A){
// CHECK-LABEL: @test_mm256_cvttsps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
  return _mm256_cvttsps_epu64( A);
}

__m256i test_mm256_mask_cvttsps_epu64(__m256i W,__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_mask_cvttsps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
    return _mm256_mask_cvttsps_epu64(W,U, A);
}

__m256i test_mm256_maskz_cvttsps_epu64(__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_maskz_cvttsps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
    return _mm256_maskz_cvttsps_epu64(U, A);
}

__m256i test_mm256_cvtts_roundps_epu64(__m128 A){
// CHECK-LABEL: @test_mm256_cvtts_roundps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
    return _mm256_cvtts_roundps_epu64(A, _MM_FROUND_NEARBYINT );
}

__m256i test_mm256_mask_cvtts_roundps_epu64(__m256i W,__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_mask_cvtts_roundps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
    return _mm256_mask_cvtts_roundps_epu64(W,U,A,_MM_FROUND_NEARBYINT );
}

__m256i test_mm256_maskz_cvtts_roundps_epu64(__mmask8 U, __m128 A){
// CHECK-LABEL: @test_mm256_maskz_cvtts_roundps_epu64
// CHECK: @llvm.x86.avx512.mask.vcvttps2uqqs.round.256(<4 x float>
    return _mm256_maskz_cvtts_roundps_epu64(U,A,_MM_FROUND_NEARBYINT );
}

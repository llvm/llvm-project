// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -emit-llvm -o - | FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512

#include <immintrin.h>

__m128i test_mm_gf2p8affineinv_epi64_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: @test_mm_gf2p8affineinv_epi64_epi8
  // SSE: @llvm.x86.vgf2p8affineinvqb.128
  return _mm_gf2p8affineinv_epi64_epi8(A, B, 1);
}

__m128i test_mm_gf2p8affine_epi64_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: @test_mm_gf2p8affine_epi64_epi8
  // SSE: @llvm.x86.vgf2p8affineqb.128
  return _mm_gf2p8affine_epi64_epi8(A, B, 1);
}

__m128i test_mm_gf2p8mul_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: @test_mm_gf2p8mul_epi8
  // SSE: @llvm.x86.vgf2p8mulb.128
  return _mm_gf2p8mul_epi8(A, B);
}

#ifdef __AVX__
__m256i test_mm256_gf2p8affineinv_epi64_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: @test_mm256_gf2p8affineinv_epi64_epi8
  // AVX: @llvm.x86.vgf2p8affineinvqb.256
  return _mm256_gf2p8affineinv_epi64_epi8(A, B, 1);
}

__m256i test_mm256_gf2p8affine_epi64_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: @test_mm256_gf2p8affine_epi64_epi8
  // AVX: @llvm.x86.vgf2p8affineqb.256
  return _mm256_gf2p8affine_epi64_epi8(A, B, 1);
}

__m256i test_mm256_gf2p8mul_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: @test_mm256_gf2p8mul_epi8
  // AVX: @llvm.x86.vgf2p8mulb.256
  return _mm256_gf2p8mul_epi8(A, B);
}
#endif // __AVX__

#ifdef __AVX512BW__
__m512i test_mm512_gf2p8affineinv_epi64_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.512
  return _mm512_gf2p8affineinv_epi64_epi8(A, B, 1);
}

__m512i test_mm512_mask_gf2p8affineinv_epi64_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_mask_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m512i test_mm512_maskz_gf2p8affineinv_epi64_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_maskz_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m256i test_mm256_mask_gf2p8affineinv_epi64_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_mask_gf2p8affineinv_epi64_epi8
  // AVX256: @llvm.x86.vgf2p8affineinvqb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m256i test_mm256_maskz_gf2p8affineinv_epi64_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_maskz_gf2p8affineinv_epi64_epi8
  // AVX256: @llvm.x86.vgf2p8affineinvqb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m128i test_mm_mask_gf2p8affineinv_epi64_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512-LABEL: @test_mm_mask_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.128
  // AVX512: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m128i test_mm_maskz_gf2p8affineinv_epi64_epi8(__mmask16 U, __m128i A, __m128i B) {
  // AVX512-LABEL: @test_mm_maskz_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.128
  // AVX512: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m512i test_mm512_gf2p8affine_epi64_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.512
  return _mm512_gf2p8affine_epi64_epi8(A, B, 1);
}

__m512i test_mm512_mask_gf2p8affine_epi64_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_mask_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m512i test_mm512_maskz_gf2p8affine_epi64_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_maskz_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m256i test_mm256_mask_gf2p8affine_epi64_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_mask_gf2p8affine_epi64_epi8
  // AVX256: @llvm.x86.vgf2p8affineqb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m256i test_mm256_maskz_gf2p8affine_epi64_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_maskz_gf2p8affine_epi64_epi8
  // AVX256: @llvm.x86.vgf2p8affineqb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m128i test_mm_mask_gf2p8affine_epi64_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512-LABEL: @test_mm_mask_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.128
  // AVX512: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m128i test_mm_maskz_gf2p8affine_epi64_epi8(__mmask16 U, __m128i A, __m128i B) {
  // AVX512-LABEL: @test_mm_maskz_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.128
  // AVX512: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m512i test_mm512_gf2p8mul_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_gf2p8mul_epi8
  // AVX512: @llvm.x86.vgf2p8mulb.512
  return _mm512_gf2p8mul_epi8(A, B);
}

__m512i test_mm512_mask_gf2p8mul_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_mask_gf2p8mul_epi8
  // AVX512: @llvm.x86.vgf2p8mulb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8mul_epi8(S, U, A, B);
}

__m512i test_mm512_maskz_gf2p8mul_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512-LABEL: @test_mm512_maskz_gf2p8mul_epi8
  // AVX512: @llvm.x86.vgf2p8mulb.512
  // AVX512: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8mul_epi8(U, A, B);
}

__m256i test_mm256_mask_gf2p8mul_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_mask_gf2p8mul_epi8
  // AVX256: @llvm.x86.vgf2p8mulb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8mul_epi8(S, U, A, B);
}

__m256i test_mm256_maskz_gf2p8mul_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX256-LABEL: @test_mm256_maskz_gf2p8mul_epi8
  // AVX256: @llvm.x86.vgf2p8mulb.256
  // AVX256: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8mul_epi8(U, A, B);
}

__m128i test_mm_mask_gf2p8mul_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512-LABEL: @test_mm_mask_gf2p8mul_epi8
  // AVX512: @llvm.x86.vgf2p8mulb.128
  // AVX512: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8mul_epi8(S, U, A, B);
}
#endif // __AVX512BW__

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -emit-llvm -o - | FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512,AVX512BW

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -emit-llvm -o - | FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512,AVX512BW

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512f -target-feature +avx512vl -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +gfni -target-feature +avx512bw -target-feature +avx512vl -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck %s --check-prefixes SSE,AVX,AVX512,AVX512BW

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_gf2p8affineinv_epi64_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: test_mm_gf2p8affineinv_epi64_epi8
  // SSE: @llvm.x86.vgf2p8affineinvqb.128
  return _mm_gf2p8affineinv_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v16qu(
    _mm_gf2p8affineinv_epi64_epi8(
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        _mm_set_epi64x(0x0102040810204080ULL, 0x0102040810204080ULL), 0x63),
    164, 134, 130, 211, 163, 74, 44, 139, 178, 24, 49, 168, 149, 238, 98, 99));

__m128i test_mm_gf2p8affine_epi64_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: test_mm_gf2p8affine_epi64_epi8
  // SSE: @llvm.x86.vgf2p8affineqb.128
  return _mm_gf2p8affine_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v16qu(
    _mm_gf2p8affine_epi64_epi8(
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        _mm_set_epi64x(0x0102040810204080ULL, 0x0102040810204080ULL), 0x63),
    108, 109, 110, 111, 104, 105, 106, 107, 100, 101, 102, 103, 96, 97, 98, 99));

__m128i test_mm_gf2p8mul_epi8(__m128i A, __m128i B) {
  // SSE-LABEL: test_mm_gf2p8mul_epi8
  // SSE: @llvm.x86.vgf2p8mulb.128
  return _mm_gf2p8mul_epi8(A, B);
}
TEST_CONSTEXPR(match_v16qu(
    _mm_gf2p8mul_epi8(_mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                      _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
    0, 14, 26, 20, 44, 34, 54, 56, 56, 54, 34, 44, 20, 26, 14, 0));

#ifdef __AVX__
__m256i test_mm256_gf2p8affineinv_epi64_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: test_mm256_gf2p8affineinv_epi64_epi8
  // AVX: @llvm.x86.vgf2p8affineinvqb.256
  return _mm256_gf2p8affineinv_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v32qu(
    _mm256_gf2p8affineinv_epi64_epi8(
        _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
        _mm256_set_epi64x(0x0102040810204080ULL, 0x0102040810204080ULL,
                          0x0102040810204080ULL, 0x0102040810204080ULL),
        0x63),
    209, 141, 35, 156, 175, 158, 92, 59, 60, 3, 72, 250, 40, 201, 215, 23, 164,
    134, 130, 211, 163, 74, 44, 139, 178, 24, 49, 168, 149, 238, 98, 99));

__m256i test_mm256_gf2p8affine_epi64_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: test_mm256_gf2p8affine_epi64_epi8
  // AVX: @llvm.x86.vgf2p8affineqb.256
  return _mm256_gf2p8affine_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v32qu(
    _mm256_gf2p8affine_epi64_epi8(
        _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
        _mm256_set_epi64x(0x0102040810204080ULL, 0x0102040810204080ULL,
                          0x0102040810204080ULL, 0x0102040810204080ULL),
        0x63),
    124, 125, 126, 127, 120, 121, 122, 123, 116, 117, 118, 119, 112, 113, 114,
    115, 108, 109, 110, 111, 104, 105, 106, 107, 100, 101, 102, 103, 96, 97, 98,
    99));

__m256i test_mm256_gf2p8mul_epi8(__m256i A, __m256i B) {
  // AVX-LABEL: test_mm256_gf2p8mul_epi8
  // AVX: @llvm.x86.vgf2p8mulb.256
  return _mm256_gf2p8mul_epi8(A, B);
}
TEST_CONSTEXPR(match_v32qu(
    _mm256_gf2p8mul_epi8(
        _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
        _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
    0, 30, 58, 36, 108, 114, 86, 72, 184, 166, 130, 156, 212, 202, 238, 240, 240,
    238, 202, 212, 156, 130, 166, 184, 72, 86, 114, 108, 36, 58, 30, 0));
#endif // __AVX__

#ifdef __AVX512F__
__m512i test_mm512_gf2p8affineinv_epi64_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: test_mm512_gf2p8affineinv_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineinvqb.512
  return _mm512_gf2p8affineinv_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v64qu(
    _mm512_gf2p8affineinv_epi64_epi8(
        _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63),
        _mm512_set_epi64(0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL),
        0x63),
    122, 58, 216, 20, 12, 67, 86, 145, 33, 5, 90, 144, 15, 241, 38, 79, 161,
    193, 39, 83, 118, 251, 105, 162, 170, 203, 46, 54, 146, 57, 13, 89, 209, 141,
    35, 156, 175, 158, 92, 59, 60, 3, 72, 250, 40, 201, 215, 23, 164, 134, 130,
    211, 163, 74, 44, 139, 178, 24, 49, 168, 149, 238, 98, 99));

__m512i test_mm512_gf2p8affine_epi64_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: test_mm512_gf2p8affine_epi64_epi8
  // AVX512: @llvm.x86.vgf2p8affineqb.512
  return _mm512_gf2p8affine_epi64_epi8(A, B, 1);
}
TEST_CONSTEXPR(match_v64qu(
    _mm512_gf2p8affine_epi64_epi8(
        _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63),
        _mm512_set_epi64(0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL,
                         0x0102040810204080ULL, 0x0102040810204080ULL),
        0x63),
    92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83, 76, 77, 78,
    79, 72, 73, 74, 75, 68, 69, 70, 71, 64, 65, 66, 67, 124, 125, 126, 127, 120,
    121, 122, 123, 116, 117, 118, 119, 112, 113, 114, 115, 108, 109, 110, 111,
    104, 105, 106, 107, 100, 101, 102, 103, 96, 97, 98, 99));

__m512i test_mm512_gf2p8mul_epi8(__m512i A, __m512i B) {
  // AVX512-LABEL: test_mm512_gf2p8mul_epi8
  // AVX512: @llvm.x86.vgf2p8mulb.512
  return _mm512_gf2p8mul_epi8(A, B);
}
TEST_CONSTEXPR(match_v64qu(
    _mm512_gf2p8mul_epi8(
        _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63),
        _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
                        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,
                        33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
                        18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                        1, 0)),
    0, 62, 122, 68, 236, 210, 150, 168, 163, 157, 217, 231, 79, 113, 53, 11, 198,
    248, 188, 130, 42, 20, 80, 110, 101, 91, 31, 33, 137, 183, 243, 205, 205,
    243, 183, 137, 33, 31, 91, 101, 110, 80, 20, 42, 130, 188, 248, 198, 11, 53,
    113, 79, 231, 217, 157, 163, 168, 150, 210, 236, 68, 122, 62, 0));
#endif // __AVX512F__

#ifdef __AVX512BW__
__m512i test_mm512_mask_gf2p8affineinv_epi64_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_mask_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m512i test_mm512_maskz_gf2p8affineinv_epi64_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_maskz_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m128i test_mm_mask_gf2p8affineinv_epi64_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_mask_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m128i test_mm_maskz_gf2p8affineinv_epi64_epi8(__mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_maskz_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m256i test_mm256_mask_gf2p8affineinv_epi64_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_mask_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8affineinv_epi64_epi8(S, U, A, B, 1);
}

__m256i test_mm256_maskz_gf2p8affineinv_epi64_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_maskz_gf2p8affineinv_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineinvqb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8affineinv_epi64_epi8(U, A, B, 1);
}

__m512i test_mm512_mask_gf2p8affine_epi64_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_mask_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m512i test_mm512_maskz_gf2p8affine_epi64_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_maskz_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m128i test_mm_mask_gf2p8affine_epi64_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_mask_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m128i test_mm_maskz_gf2p8affine_epi64_epi8(__mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_maskz_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m256i test_mm256_mask_gf2p8affine_epi64_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_mask_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8affine_epi64_epi8(S, U, A, B, 1);
}

__m256i test_mm256_maskz_gf2p8affine_epi64_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_maskz_gf2p8affine_epi64_epi8
  // AVX512BW: @llvm.x86.vgf2p8affineqb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8affine_epi64_epi8(U, A, B, 1);
}

__m512i test_mm512_mask_gf2p8mul_epi8(__m512i S, __mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_mask_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_gf2p8mul_epi8(S, U, A, B);
}
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_gf2p8mul_epi8((__m512i)(__v64qi){-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
                                                  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
                                                  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
                                                  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1},
                              0xAAAAAAAAAAAAAAAAULL,
                              (__m512i)(__v64qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_gf2p8mul_epi8((__m512i)(__v64qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFFFFFFFFFFULL,
                              (__m512i)(__v64qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_gf2p8mul_epi8((__m512i)(__v64qi){42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42,
                                                  42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42,
                                                  42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42,
                                                  42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42},
                              0x0ULL,
                              (__m512i)(__v64qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_gf2p8mul_epi8((__m512i)(__v64qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFFFFFFFFFFULL,
                              (__m512i)(__v64qi){0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                                                 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                                                 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                                                 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
                              (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_gf2p8mul_epi8((__m512i)(__v64qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFFFFFFFFFFULL,
                              (__m512i)(__v64qi){0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42,
                                                 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42,
                                                 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42,
                                                 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42},
                              (__m512i)(__v64qi){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
                                                 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
                                                 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
                                                 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1}),
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42));

__m512i test_mm512_maskz_gf2p8mul_epi8(__mmask64 U, __m512i A, __m512i B) {
  // AVX512BW-LABEL: test_mm512_maskz_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.512
  // AVX512BW: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_gf2p8mul_epi8(U, A, B);
}
TEST_CONSTEXPR(match_v64qu(
    _mm512_maskz_gf2p8mul_epi8(0x5555555555555555ULL,
                               (__m512i)(__v64qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                               (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05,
    0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0,
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0,
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0));
TEST_CONSTEXPR(match_v64qu(
    _mm512_maskz_gf2p8mul_epi8(0x0ULL,
                               (__m512i)(__v64qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                               (__m512i)(__v64qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_gf2p8mul_epi8(__m128i S, __mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_mask_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_mask_gf2p8mul_epi8(S, U, A, B);
}
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_gf2p8mul_epi8((__m128i)(__v16qi){-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1},
                           0xAAAA,
                           (__m128i)(__v16qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                           (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF,
    0x05, 0xFF, 0x05));
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_gf2p8mul_epi8((__m128i)(__v16qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                           0xFFFF,
                           (__m128i)(__v16qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                           (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05));
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_gf2p8mul_epi8((__m128i)(__v16qi){42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42},
                           0x0,
                           (__m128i)(__v16qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                           (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_gf2p8mul_epi8((__m128i)(__v16qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                           0xFFFF,
                           (__m128i)(__v16qi){0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
                           (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_gf2p8mul_epi8((__m128i)(__v16qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                           0xFFFF,
                           (__m128i)(__v16qi){0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42},
                           (__m128i)(__v16qi){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1}),
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42));

__m128i test_mm_maskz_gf2p8mul_epi8(__mmask16 U, __m128i A, __m128i B) {
  // AVX512BW-LABEL: test_mm_maskz_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.128
  // AVX512BW: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm_maskz_gf2p8mul_epi8(U, A, B);
}
TEST_CONSTEXPR(match_v16qu(
    _mm_maskz_gf2p8mul_epi8(0x5555,
                            (__m128i)(__v16qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                            (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0));
TEST_CONSTEXPR(match_v16qu(
    _mm_maskz_gf2p8mul_epi8(0x0,
                            (__m128i)(__v16qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                            (__m128i)(__v16qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_gf2p8mul_epi8(__m256i S, __mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_mask_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_gf2p8mul_epi8(S, U, A, B);
}
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_gf2p8mul_epi8((__m256i)(__v32qi){-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
                                                  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1},
                              0xAAAAAAAA,
                              (__m256i)(__v32qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05,
    0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05, 0xFF, 0x05));
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_gf2p8mul_epi8((__m256i)(__v32qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFF,
                              (__m256i)(__v32qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05));
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_gf2p8mul_epi8((__m256i)(__v32qi){42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42,
                                                  42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42},
                              0x0,
                              (__m256i)(__v32qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                              (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_gf2p8mul_epi8((__m256i)(__v32qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFF,
                              (__m256i)(__v32qi){0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                                                 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
                              (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_gf2p8mul_epi8((__m256i)(__v32qi){99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
                                                  99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99},
                              0xFFFFFFFF,
                              (__m256i)(__v32qi){0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42,
                                                 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42, 0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x42},
                              (__m256i)(__v32qi){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
                                                 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1}),
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42));

__m256i test_mm256_maskz_gf2p8mul_epi8(__mmask32 U, __m256i A, __m256i B) {
  // AVX512BW-LABEL: test_mm256_maskz_gf2p8mul_epi8
  // AVX512BW: @llvm.x86.vgf2p8mulb.256
  // AVX512BW: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_gf2p8mul_epi8(U, A, B);
}
TEST_CONSTEXPR(match_v32qu(
    _mm256_maskz_gf2p8mul_epi8(0x55555555,
                               (__m256i)(__v32qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                               (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0,
    0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0, 0x05, 0));
TEST_CONSTEXPR(match_v32qu(
    _mm256_maskz_gf2p8mul_epi8(0x0,
                               (__m256i)(__v32qi){0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12,
                                                  0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12, 0x12,0x12,0x12,0x12,0x12,0x12,0x12,0x12},
                               (__m256i)(__v32qi){0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,
                                                  0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34, 0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34}),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
#endif // __AVX512BW__

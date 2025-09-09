// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_mask_compress_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_compress_epi16
  // CHECK: call <32 x i16> @llvm.x86.avx512.mask.compress.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i1> %{{.*}})
  return _mm512_mask_compress_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_maskz_compress_epi16
  // CHECK: call <32 x i16> @llvm.x86.avx512.mask.compress.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i1> %{{.*}})
  return _mm512_maskz_compress_epi16(__U, __D);
}

__m512i test_mm512_mask_compress_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_compress_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.mask.compress.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i1> %{{.*}})
  return _mm512_mask_compress_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_maskz_compress_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.mask.compress.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i1> %{{.*}})
  return _mm512_maskz_compress_epi8(__U, __D);
}

void test_mm512_mask_compressstoreu_epi16(void *__P, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_epi16
  // CHECK: call void @llvm.masked.compressstore.v32i16(<32 x i16> %{{.*}}, ptr %{{.*}}, <32 x i1> %{{.*}})
  _mm512_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm512_mask_compressstoreu_epi8(void *__P, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_epi8
  // CHECK: call void @llvm.masked.compressstore.v64i8(<64 x i8> %{{.*}}, ptr %{{.*}}, <64 x i1> %{{.*}})
  _mm512_mask_compressstoreu_epi8(__P, __U, __D);
}

__m512i test_mm512_mask_expand_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_expand_epi16
  // CHECK: call <32 x i16> @llvm.x86.avx512.mask.expand.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i1> %{{.*}})
  return _mm512_mask_expand_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_maskz_expand_epi16
  // CHECK: call <32 x i16> @llvm.x86.avx512.mask.expand.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i1> %{{.*}})
  return _mm512_maskz_expand_epi16(__U, __D);
}

__m512i test_mm512_mask_expand_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_mask_expand_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.mask.expand.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i1> %{{.*}})
  return _mm512_mask_expand_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: test_mm512_maskz_expand_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.mask.expand.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i1> %{{.*}})
  return _mm512_maskz_expand_epi8(__U, __D);
}

__m512i test_mm512_mask_expandloadu_epi16(__m512i __S, __mmask32 __U, void const* __P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_epi16
  // CHECK: call <32 x i16> @llvm.masked.expandload.v32i16(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_mask_expandloadu_epi16(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi16(__mmask32 __U, void const* __P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_epi16
  // CHECK: call <32 x i16> @llvm.masked.expandload.v32i16(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_maskz_expandloadu_epi16(__U, __P);
}

__m512i test_mm512_mask_expandloadu_epi8(__m512i __S, __mmask64 __U, void const* __P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_epi8
  // CHECK: call <64 x i8> @llvm.masked.expandload.v64i8(ptr %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_mask_expandloadu_epi8(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi8(__mmask64 __U, void const* __P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_epi8
  // CHECK: call <64 x i8> @llvm.masked.expandload.v64i8(ptr %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_maskz_expandloadu_epi8(__U, __P);
}

__m512i test_mm512_mask_shldi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldi_epi64
  // CHECK: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 47))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldi_epi64(__S, __U, __A, __B, 47);
}
TEST_CONSTEXPR(match_v8di(_mm512_mask_shldi_epi64(((__m512i)(__v8di){ 999, 999, 999, 999, 999, 999, 999, 999}), 0x46, ((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}), 51),  999, 38280596832649216LL, -40532396646334464LL, 999, 999, 999, -47287796087390209LL, 999));

__m512i test_mm512_maskz_shldi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldi_epi64
  // CHECK: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 63))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shldi_epi64(__U, __A, __B, 63);
}
TEST_CONSTEXPR(match_v8di(_mm512_maskz_shldi_epi64(0x46, ((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}),  51),  0, 38280596832649216LL, -40532396646334464LL, 0, 0, 0, -47287796087390209LL, 0));

__m512i test_mm512_shldi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldi_epi64
  // CHECK: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 31))
  return _mm512_shldi_epi64(__A, __B, 31);
}
TEST_CONSTEXPR(match_v8di(_mm512_shldi_epi64(((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}),  51),  38280596832649215LL, 38280596832649216LL, -40532396646334464LL, 45035996273704959LL, -42784196460019713LL, 47287796087390208LL, -47287796087390209LL, 54043195528445951LL));

__m512i test_mm512_mask_shldi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldi_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 7))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldi_epi32(__S, __U, __A, __B, 7);
}
TEST_CONSTEXPR(match_v16si(_mm512_mask_shldi_epi32(((__m512i)(__v16si){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}), 0x62E9, ((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}), 11),  65536, 999, 999, 73727, 999, -73729, -75777, 81919, 999, 86015, 999, 999, 999, 94207, -92161, 999));

__m512i test_mm512_maskz_shldi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldi_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 15))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldi_epi32(__U, __A, __B, 15);
}
TEST_CONSTEXPR(match_v16si(_mm512_maskz_shldi_epi32(0x62E9, ((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}),  11),  65536, 0, 0, 73727, 0, -73729, -75777, 81919, 0, 86015, 0, 0, 0, 94207, -92161, 0));

__m512i test_mm512_shldi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldi_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 31))
  return _mm512_shldi_epi32(__A, __B, 31);
}
TEST_CONSTEXPR(match_v16si(_mm512_shldi_epi32(((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}),  11),  65536, 67584, -69632, 73727, 73728, -73729, -75777, 81919, -79873, 86015, 88063, -86017, -88065, 94207, -92161, -96256));

__m512i test_mm512_mask_shldi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldi_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 3))
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldi_epi16(__S, __U, __A, __B, 3);
}
TEST_CONSTEXPR(match_v32hi(_mm512_mask_shldi_epi16(((__m512i)(__v32hi){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}), 0x589033B5, ((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}), 7),  8319, 999, -8321, 999, 8704, -8832, 999, -8961, 9216, 9344, 999, 999, -9728, 9983, 999, 999, 999, 999, 999, 999, 10752, 999, 999, 11136, 999, 999, 999, 11775, 11903, 999, -11905, 999));

__m512i test_mm512_maskz_shldi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldi_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 7))
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldi_epi16(__U, __A, __B, 7);
}
TEST_CONSTEXPR(match_v32hi(_mm512_maskz_shldi_epi16(0x589033B5, ((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}),  7),  8319, 0, -8321, 0, 8704, -8832, 0, -8961, 9216, 9344, 0, 0, -9728, 9983, 0, 0, 0, 0, 0, 0, 10752, 0, 0, 11136, 0, 0, 0, 11775, 11903, 0, -11905, 0));

__m512i test_mm512_shldi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldi_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 15))
  return _mm512_shldi_epi16(__A, __B, 15);
}
TEST_CONSTEXPR(match_v32hi(_mm512_shldi_epi16(((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}),  7),  8319, 8320, -8321, 8576, 8704, -8832, 8960, -8961, 9216, 9344, -9472, 9727, -9728, 9983, 10111, -10112, -10240, -10241, 10623, -10497, 10752, 11007, 11008, 11136, -11137, -11392, 11647, 11775, 11903, -11777, -11905, 12160));

__m512i test_mm512_mask_shrdi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdi_epi64
  // CHECK: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 47))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shrdi_epi64(__S, __U, __A, __B, 47);
}
TEST_CONSTEXPR(match_v8di(_mm512_mask_shrdi_epi64(((__m512i)(__v8di){ 999, 999, 999, 999, 999, 999, 999, 999}), 0x46, ((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}), 51),  999, 16384, 32767, 999, 999, 999, -49153, 999));

__m512i test_mm512_maskz_shrdi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdi_epi64
  // CHECK: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 63))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shrdi_epi64(__U, __A, __B, 63);
}
TEST_CONSTEXPR(match_v8di(_mm512_maskz_shrdi_epi64(0x46, ((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}),  51),  0, 16384, 32767, 0, 0, 0, -49153, 0));

__m512i test_mm512_shrdi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdi_epi64
  // CHECK: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 31))
  return _mm512_shrdi_epi64(__A, __B, 31);
}
TEST_CONSTEXPR(match_v8di(_mm512_shrdi_epi64(((__m512i)(__v8di){ 16, 17, -18, 19, -20, 21, -22, 23}), ((__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, -8}),  51),  -8192, 16384, 32767, -32768, -32769, 49152, -49153, -65536));

__m512i test_mm512_mask_shrdi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdi_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 7))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdi_epi32(__S, __U, __A, __B, 7);
}
TEST_CONSTEXPR(match_v16si(_mm512_mask_shrdi_epi32(((__m512i)(__v16si){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}), 0x62E9, ((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}), 11),  2097152, 999, 999, -8388608, 999, -10485761, -12582913, -16777216, 999, -20971520, 999, 999, 999, -29360128, -29360129, 999));

__m512i test_mm512_maskz_shrdi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdi_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 15))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdi_epi32(__U, __A, __B, 15);
}
TEST_CONSTEXPR(match_v16si(_mm512_maskz_shrdi_epi32(0x62E9, ((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}),  11),  2097152, 0, 0, -8388608, 0, -10485761, -12582913, -16777216, 0, -20971520, 0, 0, 0, -29360128, -29360129, 0));

__m512i test_mm512_shrdi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdi_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 31))
  return _mm512_shrdi_epi32(__A, __B, 31);
}
TEST_CONSTEXPR(match_v16si(_mm512_shrdi_epi32(((__m512i)(__v16si){ 32, 33, -34, 35, 36, -37, -38, 39, -40, 41, 42, -43, -44, 45, -46, -47}), ((__m512i)(__v16si){ 1, 2, 3, -4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, 16}),  11),  2097152, 4194304, 8388607, -8388608, 10485760, -10485761, -12582913, -16777216, -16777217, -20971520, -23068672, -23068673, -25165825, -29360128, -29360129, 35651583));

__m512i test_mm512_mask_shrdi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdi_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 3))
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdi_epi16(__S, __U, __A, __B, 3);
}
TEST_CONSTEXPR(match_v32hi(_mm512_mask_shrdi_epi16(((__m512i)(__v32hi){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}), 0x589033B5, ((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}), 7),  -512, 999, -1025, 999, 2560, 3583, 999, -3585, 4608, 5120, 999, 999, 7167, -7168, 999, 999, 999, 999, 999, 999, 10752, 999, 999, 12288, 999, 999, 999, -14336, -14848, 999, -15361, 999));

__m512i test_mm512_maskz_shrdi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdi_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 15))
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shrdi_epi16(__U, __A, __B, 15);
}
TEST_CONSTEXPR(match_v32hi(_mm512_maskz_shrdi_epi16(0x589033B5, ((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}),  7),  -512, 0, -1025, 0, 2560, 3583, 0, -3585, 4608, 5120, 0, 0, 7167, -7168, 0, 0, 0, 0, 0, 0, 10752, 0, 0, 12288, 0, 0, 0, -14336, -14848, 0, -15361, 0));

__m512i test_mm512_shrdi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdi_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 31))
  return _mm512_shrdi_epi16(__A, __B, 31);
}
TEST_CONSTEXPR(match_v32hi(_mm512_shrdi_epi16(((__m512i)(__v32hi){ 64, 65, -66, 67, 68, -69, 70, -71, 72, 73, -74, 75, -76, 77, 78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, -89, 90, 91, 92, -93, -94, 95}), ((__m512i)(__v32hi){ -1, 2, -3, 4, 5, 6, 7, -8, 9, 10, 11, -12, 13, -14, -15, 16, 17, -18, -19, -20, 21, -22, 23, 24, -25, 26, -27, -28, -29, -30, -31, 32}),  7),  -512, 1024, -1025, 2048, 2560, 3583, 3584, -3585, 4608, 5120, 6143, -6144, 7167, -7168, -7680, 8703, 9215, -8705, -9728, -9729, 10752, -11264, 11776, 12288, -12289, 13823, -13824, -14336, -14848, -14849, -15361, 16384));

__m512i test_mm512_mask_shldv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_mask_shldv_epi64((__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, 0xC1, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1080863910568919041LL, 17, -18, 19, -20, 21, 91, -9223372036854775804LL));

__m512i test_mm512_maskz_shldv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shldv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_maskz_shldv_epi64(0xC1, (__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1080863910568919041LL, 0, 0, 0, 0, 0, 91, -9223372036854775804LL));

__m512i test_mm512_shldv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_shldv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_shldv_epi64((__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1080863910568919041LL, 2176, -5188146770730811392LL, 639, -3458764513820540929LL, -6917529027641081856LL, 91, -9223372036854775804LL));

__m512i test_mm512_mask_shldv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldv_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_mask_shldv_epi32((__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, 0x26D8, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 32, -33, 34, 18874367, 37748736, 37, -159383552, 327155712, -40, -5248, -1476395008, 43, 44, 360, 46, -47));

__m512i test_mm512_maskz_shldv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldv_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_maskz_shldv_epi32(0x26D8, (__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 0, 0, 0, 18874367, 37748736, 0, -159383552, 327155712, 0, -5248, -1476395008, 0, 0, 360, 0, 0));

__m512i test_mm512_shldv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldv_epi32
  // CHECK: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_shldv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_shldv_epi32((__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 2097152, -4325376, 573439, 18874367, 37748736, 77823, -159383552, 327155712, -10240, -5248, -1476395008, 1376, 719, 360, -1073741828, -2147483640));

__m512i test_mm512_mask_shldv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shldv_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_mask_shldv_epi16((__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, 0x73314D8, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), -64, 65, 66, 536, 1088, -69, 7167, 29184, -72, 73, 10240, 75, -1216, -77, -78, -79, -80, -162, 82, -83, 16385, 2751, 86, 87, -22528, 11519, 5760, -91, 92, 93, 94, 95));

__m512i test_mm512_maskz_shldv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shldv_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_maskz_shldv_epi16(0x73314D8, (__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), 0, 0, 0, 536, 1088, 0, 7167, 29184, 0, 0, 10240, 0, -1216, 0, 0, 0, -80, -162, 0, 0, 16385, 2751, 0, 0, -22528, 11519, 5760, 0, 0, 0, 0, 0));

__m512i test_mm512_shldv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shldv_epi16
  // CHECK: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_shldv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_shldv_epi16((__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), -64, 130, 267, 536, 1088, -8193, 7167, 29184, -18432, -27649, 10240, 2400, -1216, -609, -312, -32760, -80, -162, -32764, -24574, 16385, 2751, 5567, 11136, -22528, 11519, 5760, 10240, -12290, 751, 379, -16));

__m512i test_mm512_mask_shrdv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shrdv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_mask_shrdv_epi64((__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, 0xC1, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1, 17, -18, 19, -20, 21, 4611686018427387909LL, 17));

__m512i test_mm512_maskz_shrdv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shrdv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_maskz_shrdv_epi64(0xC1, (__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1, 0, 0, 0, 0, 0, 4611686018427387909LL, 17));

__m512i test_mm512_shrdv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdv_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_shrdv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8di(_mm512_shrdv_epi64((__m512i)(__v8di){ -16, 17, -18, 19, -20, 21, 22, -23}, (__m512i)(__v8di){ -1, 2, 3, -4, -5, 6, -7, 8}, (__m512i)(__v8di){ -8, 7, -6, 5, -4, -3, 2, -1}), -1, 288230376151711744LL, 255, -2305843009213693952LL, -65, 48, 4611686018427387909LL, 17));

__m512i test_mm512_mask_shrdv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdv_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_mask_shrdv_epi32((__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, 0x26D8, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 32, -33, 34, -32768, 20480, 37, 8191, 4096, -40, 369098751, 704, 43, 44, -1073741819, 46, -47));

__m512i test_mm512_maskz_shrdv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdv_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_maskz_shrdv_epi32(0x26D8, (__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 0, 0, 0, -32768, 20480, 0, 8191, 4096, 0, 369098751, 704, 0, 0, -1073741819, 0, 0));

__m512i test_mm512_shrdv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdv_epi32
  // CHECK: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_shrdv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v16si(_mm512_shrdv_epi32((__m512i)(__v16si){ 32, -33, 34, 35, 36, 37, -38, 39, -40, -41, 42, 43, 44, 45, 46, -47}, (__m512i)(__v16si){ 1, 2, -3, -4, 5, -6, 7, 8, 9, 10, 11, 12, -13, 14, -15, 16}, (__m512i)(__v16si){ 16, -15, 14, -13, -12, 11, -10, -9, 8, 7, -6, 5, 4, 3, -2, -1}), 65536, 98303, -786432, -32768, 20480, -12582912, 8191, 4096, 167772159, 369098751, 704, 1610612737, 805306370, -1073741819, -60, 33));

__m512i test_mm512_mask_shrdv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shrdv_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_mask_shrdv_epi16((__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, 0x73314D8, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), -64, 65, 66, -32760, 20484, -69, -448, 1151, -72, 73, 704, 75, -8197, -77, -78, -79, -80, 32727, 82, -83, 336, 20482, 86, 87, 6655, -13312, 27649, -91, 92, 93, 94, 95));

__m512i test_mm512_maskz_shrdv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shrdv_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shrdv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_maskz_shrdv_epi16(0x73314D8, (__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), 0, 0, 0, -32760, 20484, 0, -448, 1151, 0, 0, 704, 0, -8197, 0, 0, 0, -80, 32727, 0, 0, 336, 20482, 0, 0, 6655, -13312, 27649, 0, 0, 0, 0, 0));

__m512i test_mm512_shrdv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shrdv_epi16
  // CHECK: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_shrdv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_shrdv_epi16((__m512i)(__v32hi){ -64, 65, 66, 67, 68, -69, 70, -71, -72, 73, 74, 75, -76, -77, -78, -79, -80, -81, 82, -83, 84, 85, 86, 87, -88, 89, 90, -91, 92, 93, 94, 95}, (__m512i)(__v32hi){ -1, 2, -3, 4, 5, -6, -7, 8, 9, -10, 11, 12, 13, -14, 15, 16, -17, 18, 19, 20, 21, -22, -23, 24, 25, -26, 27, 28, -29, -30, -31, -32}, (__m512i)(__v32hi){ -32, -31, -30, -29, -28, 27, 26, 25, 24, -23, -22, 21, 20, 19, 18, -17, -16, -15, 14, 13, 12, -11, -10, -9, -8, 7, 6, -5, -4, 3, 2, -1}), -64, 32, 16400, -32760, 20484, -161, -448, 1151, 2559, -1280, 704, 24578, -8197, 24566, -20, 33, -80, 32727, 76, 167, 336, 20482, -23551, 12288, 6655, -13312, 27649, 927, -464, 16395, 16407, -64));


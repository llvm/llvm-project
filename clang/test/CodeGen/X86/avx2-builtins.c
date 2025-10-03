// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X86

#include <immintrin.h>
#include "builtin_test_helpers.h"

// NOTE: This should match the tests in llvm/test/CodeGen/X86/avx2-intrinsics-fast-isel.ll

__m256i test_mm256_abs_epi8(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi8
  // CHECK: [[ABS:%.*]] = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %{{.*}}, i1 false)
  return _mm256_abs_epi8(a);
}
TEST_CONSTEXPR(match_v32qi(_mm256_abs_epi8((__m256i)(__v32qs){0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +100, +50, -100, +20, +80, -50, +120, -20, -100, -50, +100, -20, -80, +50, -120, +20}), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100, 50, 100, 20, 80, 50, 120, 20, 100, 50, 100, 20, 80, 50, 120, 20));

__m256i test_mm256_abs_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi16
  // CHECK: [[ABS:%.*]] = call <16 x i16> @llvm.abs.v16i16(<16 x i16> %{{.*}}, i1 false)
  return _mm256_abs_epi16(a);
}
TEST_CONSTEXPR(match_v16hi(_mm256_abs_epi16((__m256i)(__v16hi){+5, -3, -32767, +32767, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129}), 5, 3, 32767, 32767, 10, 8, 0, 256, 256, 128, 3, 9, 15, 33, 63, 129));

__m256i test_mm256_abs_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi32
  // CHECK: [[ABS:%.*]] = call <8 x i32> @llvm.abs.v8i32(<8 x i32> %{{.*}}, i1 false)
  return _mm256_abs_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_abs_epi32((__m256i)(__v8si){+5, -3, -2147483647, +2147483647, 0, -256, +256, +1025}), 5, 3, 2147483647, 2147483647, 0, 256, 256, 1025));

__m256i test_mm256_add_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi8
  // CHECK: add <32 x i8>
  return _mm256_add_epi8(a, b);
}

TEST_CONSTEXPR(match_v32qi(_mm256_add_epi8((__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}),  -65, -67, 69, 63, 73, -75, -63, -63, -63, 63, 85, -87, 89, 91, -63, 95, -97, 99, 101, -63, 63, -63, 63, 63, -63, -63, -117, 63, -121, 123, 63, 127));

__m256i test_mm256_add_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi16
  // CHECK: add <16 x i16>
  return _mm256_add_epi16(a, b);
}

TEST_CONSTEXPR(match_v16hi(_mm256_add_epi16((__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}),  -33, -31, 37, -39, -31, 31, -31, -31, -31, -31, -53, 31, -57, -31, 31, 31));

__m256i test_mm256_add_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi32
  // CHECK: add <8 x i32>
  return _mm256_add_epi32(a, b);
}

TEST_CONSTEXPR(match_v8si(_mm256_add_epi32((__m256i)(__v8si){ 16, 17, 18, -19, 20, -21, 22, 23}, (__m256i)(__v8si){ -1, 2, 3, 4, 5, 6, 7, 8}),  15, 19, 21, -15, 25, -15, 29, 31));

__m256i test_mm256_add_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi64
  // CHECK: add <4 x i64>
  return _mm256_add_epi64(a, b);
}

TEST_CONSTEXPR(match_v4di(_mm256_add_epi64((__m256i)(__v4di){ 8, -9, 10, 11}, (__m256i)(__v4di){ -1, -2, 3, 4}),  7, -11, 13, 15));

__m256i test_mm256_adds_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epi8
  // CHECK: call <32 x i8> @llvm.sadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_adds_epi8(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_adds_epi8((__m256i)(__v32qs){0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +100, +50, -100, +20, +80, -50, +120, -20, -100, -50, +100, -20, -80, +50, -120, +20}, (__m256i)(__v32qs){0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +50, +80, -50, +110, +60, -30, +20, -10, +50, +80, -50, +110, +60, -30, +20, -10}), 0, +2, +4, +6, +8, +10, +12, +14, +16, +18, +20, +22, +24, +26, +28, +30, +127, +127, -128, +127, +127, -80, +127, -30, -50, +30, +50, +90, -20, +20, -100, +10));

__m256i test_mm256_adds_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epi16
  // CHECK: call <16 x i16> @llvm.sadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_adds_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_adds_epi16((__m256i)(__v16hi){0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, +32000, -32000, +32000, -32000}, (__m256i)(__v16hi){0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, +800, -800, -800, +800}), 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, +32767, -32768, +31200, -31200));

__m256i test_mm256_adds_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epu8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_adds_epu8(a, b);
}
TEST_CONSTEXPR(match_v32qu(_mm256_adds_epu8((__m256i)(__v32qu){0, 0, 0, 0, +64, +64, +64, +64, +64, +64, +127, +127, +127, +127, +127, +127, +128, +128, +128, +128, +128, +128, +192, +192, +192, +192, +192, +192, +255, +255, +255, +255}, (__m256i)(__v32qu){0, +127, +128, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +127, +128, +255}), 0, +127, +128, +255, +64, +128, +191, +192, +255, +255, +127, +191, +254, +255, +255, +255, +128, +192, +255, +255, +255, +255, +192, +255, +255, +255, +255, +255, +255, +255, +255, +255));

__m256i test_mm256_adds_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epu16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_adds_epu16(a, b);
}
TEST_CONSTEXPR(match_v16hu(_mm256_adds_epu16((__m256i)(__v16hu){0, 0, 0, 0, +32767, +32767, +32767, +32767, +32768, +32768, +32768, +32768, +65535, +65535, +65535, +65535}, (__m256i)(__v16hu){0, +32767, +32768, +65535, 0, +32767, +32768, +65535, 0, +32767, +32768, +65535, 0, +32767, +32768, +65535}), 0, +32767, +32768, +65535, +32767, +65534, +65535, +65535, +32768, +65535, +65535, +65535, +65535, +65535, +65535, +65535));

__m256i test_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  return _mm256_alignr_epi8(a, b, 2);
}

__m256i test2_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test2_mm256_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  return _mm256_alignr_epi8(a, b, 17);
}

__m256i test_mm256_and_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_and_si256
  // CHECK: and <4 x i64>
  return _mm256_and_si256(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_and_si256((__m256i)(__v4di){0, -1, 0, -1}, (__m256i)(__v4di){0, 0, -1, -1}), 0, 0, 0, -1));

__m256i test_mm256_andnot_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_andnot_si256
  // CHECK: xor <4 x i64>
  // CHECK: and <4 x i64>
  return _mm256_andnot_si256(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_andnot_si256((__m256i)(__v4di){0, -1, 0, -1}, (__m256i)(__v4di){0, 0, -1, -1}), 0, 0, -1, 0));

__m256i test_mm256_avg_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_avg_epu8
  // CHECK: call <32 x i8> @llvm.x86.avx2.pavg.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_avg_epu8(a, b);
}
TEST_CONSTEXPR(match_v32qu(_mm256_avg_epu8((__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32));

__m256i test_mm256_avg_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_avg_epu16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pavg.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_avg_epu16(a, b);
}
TEST_CONSTEXPR(match_v16hu(_mm256_avg_epu16((__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

// FIXME: We should also lower the __builtin_ia32_pblendw128 (and similar)
// functions to this IR. In the future we could delete the corresponding
// intrinsic in LLVM if it's not being used anymore.
__m256i test_mm256_blend_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi16
  // CHECK-NOT: @llvm.x86.avx2.pblendw
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 17, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 25, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_blend_epi16(a, b, 2);
}
TEST_CONSTEXPR(match_v16hi(_mm256_blend_epi16(((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}), ((__m256i)(__v16hi){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}), 0x00), 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
TEST_CONSTEXPR(match_v16hi(_mm256_blend_epi16(((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}), ((__m256i)(__v16hi){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}), 0x5A), 1,-2,3,-4,-5,6,-7,8,9,-10,11,-12,-13,14,-15,16));
TEST_CONSTEXPR(match_v16hi(_mm256_blend_epi16(((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}), ((__m256i)(__v16hi){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}), 0x94), 1,2,-3,4,-5,6,7,-8,9,10,-11,12,-13,14,15,-16));
TEST_CONSTEXPR(match_v16hi(_mm256_blend_epi16(((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}), ((__m256i)(__v16hi){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}), 0xFF), -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16));

__m128i test_mm_blend_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm_blend_epi32(a, b, 0x05);
}
TEST_CONSTEXPR(match_v4si(_mm_blend_epi32(((__m128i)(__v4si){1,2,3,4}), ((__m128i)(__v4si){-1,-2,-3,-4}), 0x0), 1,2,3,4));
TEST_CONSTEXPR(match_v4si(_mm_blend_epi32(((__m128i)(__v4si){1,2,3,4}), ((__m128i)(__v4si){-1,-2,-3,-4}), 0x5), -1,2,-3,4));
TEST_CONSTEXPR(match_v4si(_mm_blend_epi32(((__m128i)(__v4si){1,2,3,4}), ((__m128i)(__v4si){-1,-2,-3,-4}), 0xA), 1,-2,3,-4));
TEST_CONSTEXPR(match_v4si(_mm_blend_epi32(((__m128i)(__v4si){1,2,3,4}), ((__m128i)(__v4si){-1,-2,-3,-4}), 0xF), -1,-2,-3,-4));

__m256i test_mm256_blend_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.256
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_epi32(a, b, 0x35);
}
TEST_CONSTEXPR(match_v8si(_mm256_blend_epi32(((__m256i)(__v8si){1,2,3,4,5,6,7,8}), ((__m256i)(__v8si){-1,-2,-3,-4,-5,-6,-7,-8}), 0x00), 1,2,3,4,5,6,7,8));
TEST_CONSTEXPR(match_v8si(_mm256_blend_epi32(((__m256i)(__v8si){1,2,3,4,5,6,7,8}), ((__m256i)(__v8si){-1,-2,-3,-4,-5,-6,-7,-8}), 0xA5), -1,2,-3,4,5,-6,7,-8));
TEST_CONSTEXPR(match_v8si(_mm256_blend_epi32(((__m256i)(__v8si){1,2,3,4,5,6,7,8}), ((__m256i)(__v8si){-1,-2,-3,-4,-5,-6,-7,-8}), 0x94), 1,2,-3,4,-5,6,7,-8));
TEST_CONSTEXPR(match_v8si(_mm256_blend_epi32(((__m256i)(__v8si){1,2,3,4,5,6,7,8}), ((__m256i)(__v8si){-1,-2,-3,-4,-5,-6,-7,-8}), 0xFF), -1,-2,-3,-4,-5,-6,-7,-8));

__m256i test_mm256_blendv_epi8(__m256i a, __m256i b, __m256i m) {
  // CHECK-LABEL: test_mm256_blendv_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_blendv_epi8(a, b, m);
}
TEST_CONSTEXPR(match_v32qi(_mm256_blendv_epi8((__m256i)(__v32qs){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},(__m256i)(__v32qs){-90,-91,-92,-93,-94,-95,-96,-97,-98,-99,-100,-101,-12,-13,-104,-105,-106,-107,-108,-109,-100,-101,-12,-13,-104,-105,-106,-107,-108,-109,-120,-121},(__m256i)(__v32qs){0,0,0,-1,0,-1,-1,0,0,0,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,-1}), 0, 1, 2, -93, 4, -95, -96, 7, 8, 9, -100, -101, 12, -13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -13, -104, -105, 26, 27, 28, 29, 30, -121));

__m128i test_mm_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.128
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  return _mm_broadcastb_epi8(a);
}
TEST_CONSTEXPR(match_v16qi(_mm_broadcastb_epi8((__m128i)(__v16qi){42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));

__m256i test_mm256_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.256
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  return _mm256_broadcastb_epi8(a);
}
TEST_CONSTEXPR(match_v32qi(_mm256_broadcastb_epi8((__m128i)(__v16qi){42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));

__m128i test_mm_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  return _mm_broadcastd_epi32(a);
}
TEST_CONSTEXPR(match_v4si(_mm_broadcastd_epi32((__m128i)(__v4si){-42, 0, 0, 0}), -42, -42, -42, -42));

__m256i test_mm256_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.256
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  return _mm256_broadcastd_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_broadcastd_epi32((__m128i)(__v4si){-42, 0, 0, 0}), -42, -42, -42, -42, -42, -42, -42, -42));

__m128i test_mm_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.128
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  return _mm_broadcastq_epi64(a);
}
TEST_CONSTEXPR(match_v2di(_mm_broadcastq_epi64((__m128i)(__v2di){-42, 0}), -42, -42));

__m256i test_mm256_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  return _mm256_broadcastq_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_broadcastq_epi64((__m128i)(__v2di){-42, 0}), -42, -42, -42, -42));

__m128d test_mm_broadcastsd_pd(__m128d a) {
  // CHECK-LABEL: test_mm_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  return _mm_broadcastsd_pd(a);
}
TEST_CONSTEXPR(match_m128d(_mm_broadcastsd_pd((__m128d){+7.0, -7.0}), +7.0, +7.0));

__m256d test_mm256_broadcastsd_pd(__m128d a) {
  // CHECK-LABEL: test_mm256_broadcastsd_pd
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.sd.pd.256
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  return _mm256_broadcastsd_pd(a);
}
TEST_CONSTEXPR(match_m256d(_mm256_broadcastsd_pd((__m128d){+7.0, -7.0}), +7.0, +7.0, +7.0, +7.0));

__m256i test_mm256_broadcastsi128_si256(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastsi128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcastsi128_si256(a);
}
TEST_CONSTEXPR(match_m256i(_mm256_broadcastsi128_si256((__m128i)(__v2di){3, 45}), 3, 45, 3, 45));

__m256i test_mm_broadcastsi128_si256(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastsi128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm_broadcastsi128_si256(a);
}
TEST_CONSTEXPR(match_m256i(_mm_broadcastsi128_si256(((__m128i)(__v2di){42, -99})), 42, -99, 42, -99));

__m128 test_mm_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  return _mm_broadcastss_ps(a);
}
TEST_CONSTEXPR(match_m128(_mm_broadcastss_ps((__m128){-4.0f, +5.0f, +6.0f, +7.0f}), -4.0f, -4.0f, -4.0f, -4.0f));

__m256 test_mm256_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm256_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps.256
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  return _mm256_broadcastss_ps(a);
}
TEST_CONSTEXPR(match_m256(_mm256_broadcastss_ps((__m128){-4.0f, +5.0f, +6.0f, +7.0f}), -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -4.0f));

__m128i test_mm_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.128
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  return _mm_broadcastw_epi16(a);
}
TEST_CONSTEXPR(match_v8hi(_mm_broadcastw_epi16((__m128i)(__v8hi){42, 0, 0, 0, 0, 0, 0, 0}), 42, 42, 42, 42, 42, 42, 42, 42));

__m256i test_mm256_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.256
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  return _mm256_broadcastw_epi16(a);
}
TEST_CONSTEXPR(match_v16hi(_mm256_broadcastw_epi16((__m128i)(__v8hi){42, 0, 0, 0, 0, 0, 0, 0}), 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42));

__m256i test_mm256_bslli_epi128(__m256i a) {
  // CHECK-LABEL: test_mm256_bslli_epi128
  // CHECK: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  return _mm256_bslli_epi128(a, 3);
}

__m256i test_mm256_bsrli_epi128(__m256i a) {
  // CHECK-LABEL: test_mm256_bsrli_epi128
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  return _mm256_bsrli_epi128(a, 3);
}

__m256i test_mm256_cmpeq_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi8
  // CHECK: icmp eq <32 x i8>
  return _mm256_cmpeq_epi8(a, b);
}
TEST_CONSTEXPR(match_v16qi(_mm_cmpeq_epi8(
    (__m128i)(__v16qs){1,-2,3,-4,-5,6,-7,8,-9,10,-11,12,-13,14,-15,16},
    (__m128i)(__v16qs){10,-2,6,-4,-5,12,-14,8,-9,20,-22,12,-26,14,-30,16}),
    0,-1,0,-1,-1,0,0,-1,-1,0,0,-1,0,-1,0,-1));

__m256i test_mm256_cmpeq_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi16
  // CHECK: icmp eq <16 x i16>
  return _mm256_cmpeq_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_cmpeq_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-10, -2, +6, -4, +5, -12, +14, -8, +9, -20, +22, -12, +26, -14, +30, -16}), 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1));

__m256i test_mm256_cmpeq_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi32
  // CHECK: icmp eq <8 x i32>
  return _mm256_cmpeq_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_cmpeq_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-10, -2, +6, -4, +5, -12, +14, -8}), 0, -1, 0, -1, -1, 0, 0, -1));

__m256i test_mm256_cmpeq_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi64
  // CHECK: icmp eq <4 x i64>
  return _mm256_cmpeq_epi64(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_cmpeq_epi64((__m256i)(__v4di){+1, -2, +3, -4}, (__m256i)(__v4di){-10, -2, +6, -4}), 0, -1, 0, -1));

__m256i test_mm256_cmpgt_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi8
  // CHECK: icmp sgt <32 x i8>
  return _mm256_cmpgt_epi8(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_cmpgt_epi8(
    (__m256i)(__v32qs){1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16},
    (__m256i)(__v32qs){10, -2, 6, -5, 30, -7, 8, -1, 20, -3, 12, -8, 25, -10, 9, -2, -10, 2, -6, 5, -30, 7, -8, 1, -20, 3, -12, 8, -25, 10, -9, 2}),
            0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1));

__m256i test_mm256_cmpgt_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi16
  // CHECK: icmp sgt <16 x i16>
  return _mm256_cmpgt_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_cmpgt_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v16hi){-10, -2, +6, -5, +30, -7, +8, -1, -10, -2, +6, -5, +30, -7, +8, -1}), -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0));

__m256i test_mm256_cmpgt_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi32
  // CHECK: icmp sgt <8 x i32>
  return _mm256_cmpgt_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_cmpgt_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-10, -2, +6, -5, +30, -7, +8, -1}), -1, 0, 0, -1, 0, -1, 0, 0));

__m256i test_mm256_cmpgt_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi64
  // CHECK: icmp sgt <4 x i64>
  return _mm256_cmpgt_epi64(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_cmpgt_epi64((__m256i)(__v4di){+1, -2, +3, -4}, (__m256i)(__v4di){-10, -2, +6, -5}), -1, 0, 0, -1));

__m256i test_mm256_cvtepi8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  return _mm256_cvtepi8_epi16(a);
}
TEST_CONSTEXPR(match_v16hi(_mm256_cvtepi8_epi16(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), -3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12));

__m256i test_mm256_cvtepi8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi32
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i32>
  return _mm256_cvtepi8_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_cvtepi8_epi32(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), -3, 2, -1, 0, 1, -2, 3, -4));

__m256i test_mm256_cvtepi8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi64
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i64>
  return _mm256_cvtepi8_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepi8_epi64(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), -3, 2, -1, 0));

__m256i test_mm256_cvtepi16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi16_epi32
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  return _mm256_cvtepi16_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_cvtepi16_epi32(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), -300, 2, -1, 0, 1, -2, 3, -4));

__m256i test_mm256_cvtepi16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi16_epi64
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i64>
  return _mm256_cvtepi16_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepi16_epi64(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), -300, 2, -1, 0));

__m256i test_mm256_cvtepi32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi32_epi64
  // CHECK: sext <4 x i32> %{{.*}} to <4 x i64>
  return _mm256_cvtepi32_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepi32_epi64(_mm_setr_epi32(-70000, 2, -1, 0)), -70000, 2, -1, 0));

__m256i test_mm256_cvtepu8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  return _mm256_cvtepu8_epi16(a);
}
TEST_CONSTEXPR(match_v16hi(_mm256_cvtepu8_epi16(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), 253, 2, 255, 0, 1, 254, 3, 252, 5, 250, 7, 248, 9, 246, 11, 244));

__m256i test_mm256_cvtepu8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi32
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i32>
  return _mm256_cvtepu8_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_cvtepu8_epi32(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), 253, 2, 255, 0, 1, 254, 3, 252));

__m256i test_mm256_cvtepu8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi64
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i64>
  return _mm256_cvtepu8_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepu8_epi64(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12)), 253, 2, 255, 0));

__m256i test_mm256_cvtepu16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu16_epi32
  // CHECK: zext <8 x i16> {{.*}} to <8 x i32>
  return _mm256_cvtepu16_epi32(a);
}
TEST_CONSTEXPR(match_v8si(_mm256_cvtepu16_epi32(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), 65236, 2, 65535, 0, 1, 65534, 3, 65532));

__m256i test_mm256_cvtepu16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu16_epi64
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i64>
  return _mm256_cvtepu16_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepu16_epi64(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), 65236, 2, 65535, 0));

__m256i test_mm256_cvtepu32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu32_epi64
  // CHECK: zext <4 x i32> %{{.*}} to <4 x i64>
  return _mm256_cvtepu32_epi64(a);
}
TEST_CONSTEXPR(match_v4di(_mm256_cvtepu32_epi64(_mm_setr_epi32(-70000, 2, -1, 0)), 4294897296, 2, 4294967295, 0));

__m128i test0_mm256_extracti128_si256_0(__m256i a) {
  // CHECK-LABEL: test0_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <2 x i32> <i32 0, i32 1>
  return _mm256_extracti128_si256(a, 0);
}

__m128i test1_mm256_extracti128_si256_1(__m256i a) {
  // CHECK-LABEL: test1_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <2 x i32> <i32 2, i32 3>
  return _mm256_extracti128_si256(a, 1);
}

// Immediate should be truncated to one bit.
__m128i test2_mm256_extracti128_si256(__m256i a) {
  // CHECK-LABEL: test2_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <2 x i32> <i32 0, i32 1>
  return _mm256_extracti128_si256(a, 0);
}

__m256i test_mm256_hadd_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadd_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.phadd.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hadd_epi16(a, b);
}

__m256i test_mm256_hadd_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.phadd.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_hadd_epi32(a, b);
}

__m256i test_mm256_hadds_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadds_epi16
  // CHECK:call <16 x i16> @llvm.x86.avx2.phadd.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hadds_epi16(a, b);
}

__m256i test_mm256_hsub_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsub_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.phsub.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hsub_epi16(a, b);
}

__m256i test_mm256_hsub_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsub_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.phsub.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_hsub_epi32(a, b);
}

__m256i test_mm256_hsubs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsubs_epi16
  // CHECK:call <16 x i16> @llvm.x86.avx2.phsub.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hsubs_epi16(a, b);
}

__m128i test_mm_i32gather_epi32(int const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_i32gather_epi32(b, c, 2);
}

__m128i test_mm_mask_i32gather_epi32(__m128i a, int const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i32gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_mask_i32gather_epi32(a, b, c, d, 2);
}

__m256i test_mm256_i32gather_epi32(int const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i32gather_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> %{{.*}}, ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i8 2)
  return _mm256_i32gather_epi32(b, c, 2);
}

__m256i test_mm256_mask_i32gather_epi32(__m256i a, int const *b, __m256i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> %{{.*}}, ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_i32gather_epi64(long long const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.gather.d.q(<2 x i64> zeroinitializer, ptr %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_i32gather_epi64(b, c, 2);
}

__m128i test_mm_mask_i32gather_epi64(__m128i a, long long const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i32gather_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.gather.d.q(<2 x i64> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_mask_i32gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_i32gather_epi64(long long const *b, __m128i c) {
  // X64-LABEL: test_mm256_i32gather_epi64
  // X64: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> zeroinitializer, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i32gather_epi64
  // X86: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_i32gather_epi64(b, c, 2);
}

__m256i test_mm256_mask_i32gather_epi64(__m256i a, long long const *b, __m128i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_epi64(a, b, c, d, 2);
}

__m128d test_mm_i32gather_pd(double const *b, __m128i c) {
  // X64-LABEL: test_mm_i32gather_pd
  // X64:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // X64-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // X64-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // X64: call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> zeroinitializer, ptr %{{.*}}, <4 x i32> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm_i32gather_pd
  // X86:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // X86-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // X86-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // X86: call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_i32gather_pd(b, c, 2);
}

__m128d test_mm_mask_i32gather_pd(__m128d a, double const *b, __m128i c, __m128d d) {
  // CHECK-LABEL: test_mm_mask_i32gather_pd
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_mask_i32gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_i32gather_pd(double const *b, __m128i c) {
  // X64-LABEL: test_mm256_i32gather_pd
  // X64:         [[CMP:%.*]] = fcmp oeq <4 x double>
  // X64-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i64>
  // X64-NEXT:    [[BC:%.*]] = bitcast <4 x i64> [[SEXT]] to <4 x double>
  // X64: call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> zeroinitializer, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i32gather_pd
  // X86:         [[CMP:%.*]] = fcmp oeq <4 x double>
  // X86-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i64>
  // X86-NEXT:    [[BC:%.*]] = bitcast <4 x i64> [[SEXT]] to <4 x double>
  // X86: call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_i32gather_pd(b, c, 2);
}

__m256d test_mm256_mask_i32gather_pd(__m256d a, double const *b, __m128i c, __m256d d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_pd
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_pd(a, b, c, d, 2);
}

__m128 test_mm_i32gather_ps(float const *b, __m128i c) {
  // X64-LABEL: test_mm_i32gather_ps
  // X64:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X64-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X64-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X64: call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> zeroinitializer, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm_i32gather_ps
  // X86:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X86-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X86-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X86: call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_i32gather_ps(b, c, 2);
}

__m128 test_mm_mask_i32gather_ps(__m128 a, float const *b, __m128i c, __m128 d) {
  // CHECK-LABEL: test_mm_mask_i32gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> %{{.*}}, ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_mask_i32gather_ps(a, b, c, d, 2);
}

__m256 test_mm256_i32gather_ps(float const *b, __m256i c) {
  // X64-LABEL: test_mm256_i32gather_ps
  // X64:         [[CMP:%.*]] = fcmp oeq <8 x float>
  // X64-NEXT:    [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i32>
  // X64-NEXT:    [[BC:%.*]] = bitcast <8 x i32> [[SEXT]] to <8 x float>
  // X64: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> zeroinitializer, ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i32gather_ps
  // X86:         [[CMP:%.*]] = fcmp oeq <8 x float>
  // X86-NEXT:    [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i32>
  // X86-NEXT:    [[BC:%.*]] = bitcast <8 x i32> [[SEXT]] to <8 x float>
  // X86: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> %{{.*}}, ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 2)
  return _mm256_i32gather_ps(b, c, 2);
}

__m256 test_mm256_mask_i32gather_ps(__m256 a, float const *b, __m256i c, __m256 d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_ps
  // CHECK: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> %{{.*}}, ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_ps(a, b, c, d, 2);
}

__m128i test_mm_i64gather_epi32(int const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d(<4 x i32> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_i64gather_epi32(b, c, 2);
}

__m128i test_mm_mask_i64gather_epi32(__m128i a, int const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d(<4 x i32> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm256_i64gather_epi32(int const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm256_i64gather_epi32(b, c, 2);
}

__m128i test_mm256_mask_i64gather_epi32(__m128i a, int const *b, __m256i c, __m128i d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_i64gather_epi64(long long const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.gather.q.q(<2 x i64> zeroinitializer, ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_i64gather_epi64(b, c, 2);
}

__m128i test_mm_mask_i64gather_epi64(__m128i a, long long const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i64gather_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.gather.q.q(<2 x i64> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_mask_i64gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_i64gather_epi64(long long const *b, __m256i c) {
  // X64-LABEL: test_mm256_i64gather_epi64
  // X64: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> zeroinitializer, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i64gather_epi64
  // X86: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_i64gather_epi64(b, c, 2);
}

__m256i test_mm256_mask_i64gather_epi64(__m256i a, long long const *b, __m256i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_epi64(a, b, c, d, 2);
}

__m128d test_mm_i64gather_pd(double const *b, __m128i c) {
  // X64-LABEL: test_mm_i64gather_pd
  // X64:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // X64-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // X64-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // X64: call <2 x double> @llvm.x86.avx2.gather.q.pd(<2 x double> zeroinitializer, ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm_i64gather_pd
  // X86:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // X86-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // X86-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // X86: call <2 x double> @llvm.x86.avx2.gather.q.pd(<2 x double> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_i64gather_pd(b, c, 2);
}

__m128d test_mm_mask_i64gather_pd(__m128d a, double const *b, __m128i c, __m128d d) {
  // CHECK-LABEL: test_mm_mask_i64gather_pd
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.q.pd(<2 x double> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_mask_i64gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_i64gather_pd(double const *b, __m256i c) {
  // X64-LABEL: test_mm256_i64gather_pd
  // X64: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  // X64: call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> zeroinitializer, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i64gather_pd
  // X86: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  // X86: call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_i64gather_pd(b, c, 2);
}

__m256d test_mm256_mask_i64gather_pd(__m256d a, double const *b, __m256i c, __m256d d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_pd
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_pd(a, b, c, d, 2);
}

__m128 test_mm_i64gather_ps(float const *b, __m128i c) {
  // X64-LABEL: test_mm_i64gather_ps
  // X64:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X64-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X64-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X64: call <4 x float> @llvm.x86.avx2.gather.q.ps(<4 x float> zeroinitializer, ptr %{{.*}}, <2 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm_i64gather_ps
  // X86:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X86-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X86-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X86: call <4 x float> @llvm.x86.avx2.gather.q.ps(<4 x float> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_i64gather_ps(b, c, 2);
}

__m128 test_mm_mask_i64gather_ps(__m128 a, float const *b, __m128i c, __m128 d) {
  // CHECK-LABEL: test_mm_mask_i64gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps(<4 x float> %{{.*}}, ptr %{{.*}}, <2 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_mask_i64gather_ps(a, b, c, d, 2);
}

__m128 test_mm256_i64gather_ps(float const *b, __m256i c) {
  // X64-LABEL: test_mm256_i64gather_ps
  // X64:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X64-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X64-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X64: call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> zeroinitializer, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  //
  // X86-LABEL: test_mm256_i64gather_ps
  // X86:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // X86-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // X86-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // X86: call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm256_i64gather_ps(b, c, 2);
}

__m128 test_mm256_mask_i64gather_ps(__m128 a, float const *b, __m256i c, __m128 d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %{{.*}}, ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_ps(a, b, c, d, 2);
}

__m256i test0_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test0_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_inserti128_si256(a, b, 0);
}
TEST_CONSTEXPR(match_m256i(_mm256_inserti128_si256(((__m256i){1ULL, 2ULL, 3ULL, 4ULL}), ((__m128i){10ULL, 20ULL}), 0), 10ULL, 20ULL, 3ULL, 4ULL));

__m256i test1_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test1_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti128_si256(a, b, 1);
}

// Immediate should be truncated to one bit.
__m256i test2_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test2_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_inserti128_si256(a, b, 0);
}

__m256i test_mm256_madd_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_madd_epi16
  // CHECK: call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_madd_epi16(a, b);
}

__m256i test_mm256_maddubs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_maddubs_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maddubs_epi16(a, b);
}

__m128i test_mm_maskload_epi32(int const *a, __m128i m) {
  // CHECK-LABEL: test_mm_maskload_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.maskload.d(ptr %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskload_epi32(a, m);
}

__m256i test_mm256_maskload_epi32(int const *a, __m256i m) {
  // CHECK-LABEL: test_mm256_maskload_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.maskload.d.256(ptr %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskload_epi32(a, m);
}

__m128i test_mm_maskload_epi64(long long const *a, __m128i m) {
  // CHECK-LABEL: test_mm_maskload_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.maskload.q(ptr %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskload_epi64(a, m);
}

__m256i test_mm256_maskload_epi64(long long const *a, __m256i m) {
  // CHECK-LABEL: test_mm256_maskload_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.maskload.q.256(ptr %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskload_epi64(a, m);
}

void test_mm_maskstore_epi32(int *a, __m128i m, __m128i b) {
  // CHECK-LABEL: test_mm_maskstore_epi32
  // CHECK: call void @llvm.x86.avx2.maskstore.d(ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  _mm_maskstore_epi32(a, m, b);
}

void test_mm256_maskstore_epi32(int *a, __m256i m, __m256i b) {
  // CHECK-LABEL: test_mm256_maskstore_epi32
  // CHECK: call void @llvm.x86.avx2.maskstore.d.256(ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  _mm256_maskstore_epi32(a, m, b);
}

void test_mm_maskstore_epi64(long long *a, __m128i m, __m128i b) {
  // CHECK-LABEL: test_mm_maskstore_epi64
  // CHECK: call void @llvm.x86.avx2.maskstore.q(ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  _mm_maskstore_epi64(a, m, b);
}

void test_mm256_maskstore_epi64(long long *a, __m256i m, __m256i b) {
  // CHECK-LABEL: test_mm256_maskstore_epi64
  // CHECK: call void @llvm.x86.avx2.maskstore.q.256(ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  _mm256_maskstore_epi64(a, m, b);
}

__m256i test_mm256_max_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi8
  // CHECK: call <32 x i8> @llvm.smax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_max_epi8(a, b);
}

TEST_CONSTEXPR(match_v32qi(_mm256_max_epi8((__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16, +17, +18, +19, +20, +21, +22, +23, +24, +25, +26, +27, +28, +29, +30, +31, +32));

__m256i test_mm256_max_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi16
  // CHECK: call <16 x i16> @llvm.smax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_max_epi16(a, b);
}

TEST_CONSTEXPR(match_v16hi(_mm256_max_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16));

__m256i test_mm256_max_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi32
  // CHECK: call <8 x i32> @llvm.smax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_max_epi32(a, b);
}

TEST_CONSTEXPR(match_v8si(_mm256_max_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-1, +2, -3, +4, -5, +6, -7, +8}), +1, +2, +3, +4, +5, +6, +7, +8));

__m256i test_mm256_max_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu8
  // CHECK: call <32 x i8> @llvm.umax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_max_epu8(a, b);
}

TEST_CONSTEXPR(match_v32qu(_mm256_max_epu8((__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32));

__m256i test_mm256_max_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu16
  // CHECK: call <16 x i16> @llvm.umax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_max_epu16(a, b);
}

TEST_CONSTEXPR(match_v16hu(_mm256_max_epu16((__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

__m256i test_mm256_max_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu32
  // CHECK: call <8 x i32> @llvm.umax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_max_epu32(a, b);
}

TEST_CONSTEXPR(match_v8su(_mm256_max_epu32((__m256i)(__v8su){1, 2, 3, 4, 5, 6, 7, 8}, (__m256i)(__v8su){0, 1, 2, 3, 4, 5, 6, 7}), 1, 2, 3, 4, 5, 6, 7, 8));

__m256i test_mm256_min_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi8
  // CHECK: call <32 x i8> @llvm.smin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_min_epi8(a, b);
}

TEST_CONSTEXPR(match_v32qi(_mm256_min_epi8((__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32));

__m256i test_mm256_min_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi16
  // CHECK: call <16 x i16> @llvm.smin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_min_epi16(a, b);
}

TEST_CONSTEXPR(match_v16hi(_mm256_min_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16));

__m256i test_mm256_min_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi32
  // CHECK: call <8 x i32> @llvm.smin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_min_epi32(a, b);
}

TEST_CONSTEXPR(match_v8si(_mm256_min_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-1, +2, -3, +4, -5, +6, -7, +8}), -1, -2, -3, -4, -5, -6, -7, -8));

__m256i test_mm256_min_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu8
  // CHECK: call <32 x i8> @llvm.umin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_min_epu8(a, b);
}

TEST_CONSTEXPR(match_v32qu(_mm256_min_epu8((__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31));

__m256i test_mm256_min_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu16
  // CHECK: call <16 x i16> @llvm.umin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_min_epu16(a, b);
}

TEST_CONSTEXPR(match_v16hu(_mm256_min_epu16((__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));

__m256i test_mm256_min_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu32
  // CHECK: call <8 x i32> @llvm.umin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_min_epu32(a, b);
}

TEST_CONSTEXPR(match_v8su(_mm256_min_epu32((__m256i)(__v8su){1, 2, 3, 4, 5, 6, 7, 8}, (__m256i)(__v8su){0, 1, 2, 3, 4, 5, 6, 7}), 0, 1, 2, 3, 4, 5, 6, 7));

int test_mm256_movemask_epi8(__m256i a) {
  // CHECK-LABEL: test_mm256_movemask_epi8
  // CHECK: call {{.*}}i32 @llvm.x86.avx2.pmovmskb(<32 x i8> %{{.*}})
  return _mm256_movemask_epi8(a);
}

__m256i test_mm256_mpsadbw_epu8(__m256i x, __m256i y) {
  // CHECK-LABEL: test_mm256_mpsadbw_epu8
  // CHECK: call <16 x i16> @llvm.x86.avx2.mpsadbw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, i8 3)
  return _mm256_mpsadbw_epu8(x, y, 3);
}

__m256i test_mm256_mul_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mul_epi32
  // CHECK: shl <4 x i64> %{{.*}}, splat (i64 32)
  // CHECK: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // CHECK: shl <4 x i64> %{{.*}}, splat (i64 32)
  // CHECK: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mul_epi32(a, b);
}
TEST_CONSTEXPR(match_m256i(_mm256_mul_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-16, -14, +12, +10, -8, +6, -4, +2}), -16, 36, -40, -28));

__m256i test_mm256_mul_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mul_epu32
  // CHECK: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // CHECK: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mul_epu32(a, b);
}
TEST_CONSTEXPR(match_m256i(_mm256_mul_epu32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-16, -14, +12, +10, -8, +6, -4, +2}), 4294967280, 36, 21474836440, 30064771044));

__m256i test_mm256_mulhi_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhi_epu16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmulhu.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhi_epu16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mulhi_epu16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), 0, -32, 0, 25, 4, -28, 0, 17, 8, -24, 0, 9, 12, 5, 14, 1));

__m256i test_mm256_mulhi_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhi_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmulh.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhi_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mulhi_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1));

__m256i test_mm256_mulhrs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhrs_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmul.hr.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhrs_epi16(a, b);
}

__m256i test_mm256_mullo_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mullo_epi16
  // CHECK: mul <16 x i16>
  return _mm256_mullo_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mullo_epi16((__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), -32, 60, 84, -104, -120, 132, 140, -144, -144, 140, 132, -120, -104, -84, -60, -32));

__m256i test_mm256_mullo_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mullo_epi32
  // CHECK: mul <8 x i32>
  return _mm256_mullo_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_mullo_epi32((__m256i)(__v8si){+1, -2, +3, -4, +5, -6, +7, -8}, (__m256i)(__v8si){-16, -14, +12, +10, -8, +6, -4, +2}), -16, 28, 36, -40, -40, -36, -28, -16));

__m256i test_mm256_or_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_or_si256
  // CHECK: or <4 x i64>
  return _mm256_or_si256(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_or_si256((__m256i)(__v4di){0, -1, 0, -1}, (__m256i)(__v4di){0, 0, -1, -1}), 0, -1, -1, -1));

__m256i test_mm256_packs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epi16
  // CHECK: call <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_packs_epi16(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_packs_epi16((__m256i)(__v16hi){130, -200, 127, -128, 300, -1000, 42, -42, 500, -500, 1, -1, 128, -129, 256, -256}, (__m256i)(__v16hi){0, 1, -1, 255, -129, 128, 20000, -32768, 32767, -32767, 127, -128, 30000, -30000, 90, -90}), 127, -128, 127, -128, 127, -128, 42, -42, 0, 1, -1, 127, -128, 127, 127, -128, 127, -128, 1, -1, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 90, -90));

__m256i test_mm256_packs_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epi32
  // CHECK: call <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_packs_epi32(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_packs_epi32((__m256i)(__v8si){40000, -50000, 32767, -32768, 70000, -70000, 42, -42}, (__m256i)(__v8si){0, 1, -1, 65536, -1000000, 1000000, 32768, -32769}), 32767, -32768, 32767, -32768, 0, 1, -1, 32767, 32767, -32768, 42, -42, -32768, 32767, 32767, -32768));

__m256i test_mm256_packs_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epu16
  // CHECK:  call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_packus_epi16(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_packus_epi16((__m256i)(__v16hi){-1, 0, 1, 127, 128, 255, 256, -200, 300, 42, -42, 500, 20000, -32768, 129, -129}, (__m256i)(__v16hi){0, 1, -1, 255, -129, 128, 20000, -32768, 32767, -32767, 127, -128, 30000, -30000, 90, -90}), 0, 0, 1, 127, -128, -1, -1, 0, 0, 1, 0, -1, 0, -128, -1, 0, -1, 42, 0, -1, -1, 0, -127, 0, -1, 0, 127, 0, -1, 0, 90, 0));

__m256i test_mm256_packs_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epu32
  // CHECK: call <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_packus_epi32(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_packus_epi32((__m256i)(__v8si){40000, -50000, 32767, -32768, 70000, -70000, 42, -42}, (__m256i)(__v8si){0, 1, -1, 65536, -1000000, 1000000, 32768, -32769}), -25536, 0, 32767, 0, 0, 1, 0, -1, -1, 0, 42, 0, 0, -1, -32768, 0));

__m256i test_mm256_permute2x128_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_permute2x128_si256
  // CHECK: shufflevector <4 x i64> zeroinitializer, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  return _mm256_permute2x128_si256(a, b, 0x38);
}

__m256i test_mm256_permute4x64_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_permute4x64_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <4 x i32> <i32 3, i32 0, i32 2, i32 0>
  return _mm256_permute4x64_epi64(a, 35);
}

__m256d test_mm256_permute4x64_pd(__m256d a) {
  // CHECK-LABEL: test_mm256_permute4x64_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 2, i32 1, i32 0>
  return _mm256_permute4x64_pd(a, 25);
}

__m256i test_mm256_permutevar8x32_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_permutevar8x32_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_permutevar8x32_epi32(a, b);
}

__m256 test_mm256_permutevar8x32_ps(__m256 a, __m256i b) {
  // CHECK-LABEL: test_mm256_permutevar8x32_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx2.permps(<8 x float> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_permutevar8x32_ps(a, b);
}

__m256i test_mm256_sad_epu8(__m256i x, __m256i y) {
  // CHECK-LABEL: test_mm256_sad_epu8
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_sad_epu8(x, y);
}

__m256i test_mm256_shuffle_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_shuffle_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_shuffle_epi8(a, b);
}

__m256i test_mm256_shuffle_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <8 x i32> <i32 3, i32 3, i32 0, i32 0, i32 7, i32 7, i32 4, i32 4>
  return _mm256_shuffle_epi32(a, 15);
}
TEST_CONSTEXPR(match_v8si(_mm256_shuffle_epi32((((__m256i)(__v8si){0,1,2,3,4,5,6,7})), 15), 3,3,0,0, 7,7,4,4));
__m256i test_mm256_shufflehi_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 6, i32 5, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 14, i32 13>
  return _mm256_shufflehi_epi16(a, 107);
}
TEST_CONSTEXPR(match_v16hi(_mm256_shufflehi_epi16((((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})), 107), 0,1,2,3, 7,6,6,5, 8,9,10,11, 15,14,14,13));
__m256i test_mm256_shufflelo_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 3, i32 0, i32 1, i32 1, i32 4, i32 5, i32 6, i32 7, i32 11, i32 8, i32 9, i32 9, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shufflelo_epi16(a, 83);
}
TEST_CONSTEXPR(match_v16hi(_mm256_shufflelo_epi16(((__m256i)(__v16hi){ 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15}), 83), 3,0,1,1, 4,5,6,7, 11,8,9,9, 12,13,14,15) );
__m256i test_mm256_sign_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_sign_epi8(a, b);
}

__m256i test_mm256_sign_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psign.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_sign_epi16(a, b);
}

__m256i test_mm256_sign_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psign.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sign_epi32(a, b);
}

__m256i test_mm256_slli_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pslli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi16(a, 3);
}
TEST_CONSTEXPR(match_v16hi(_mm256_slli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
TEST_CONSTEXPR(match_v16hi(_mm256_slli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e));
TEST_CONSTEXPR(match_v16hi(_mm256_slli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 15), 0, 0x8000, 0x0, 0x8000, 0x0, 0x8000, 0x0, 0x8000, 0x0, 0x8000, 0x0, 0x8000, 0x0, 0x8000, 0x0, 0x8000));
TEST_CONSTEXPR(match_v16hi(_mm256_slli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v16hi(_mm256_slli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 17), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_slli_epi16_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_slli_epi16_2
  // CHECK: call <16 x i16> @llvm.x86.avx2.pslli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi16(a, b);
}

__m256i test_mm256_slli_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi32(a, 3);
}
TEST_CONSTEXPR(match_v8si(_mm256_slli_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, 0), 0, 1, 2, 3, 4, 5, 6, 7));
TEST_CONSTEXPR(match_v8si(_mm256_slli_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, 1), 0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe));
TEST_CONSTEXPR(match_v8su(_mm256_slli_epi32((__m256i)(__v8su){0, 1, 2, 3, 4, 5, 6, 7}, 31), 0, 0x80000000, 0x0, 0x80000000, 0x0, 0x80000000, 0x0, 0x80000000));
TEST_CONSTEXPR(match_v8si(_mm256_slli_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, 32), 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v8si(_mm256_slli_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, 33), 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_slli_epi32_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_slli_epi32_2
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi32(a, b);
}

__m256i test_mm256_slli_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.pslli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi64(a, 3);
}
TEST_CONSTEXPR(match_v4di(_mm256_slli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 0), 0, 1, 2, 3));
TEST_CONSTEXPR(match_v4di(_mm256_slli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 1), 0, 0x2, 0x4, 0x6));
TEST_CONSTEXPR(match_v4di(_mm256_slli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 33), 0, 0x200000000LL, 0x400000000LL, 0x600000000LL));
TEST_CONSTEXPR(match_v4di(_mm256_slli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 64), 0, 0, 0, 0));
TEST_CONSTEXPR(match_v4di(_mm256_slli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 65), 0, 0, 0, 0));

__m256i test_mm256_slli_epi64_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_slli_epi64_2
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.pslli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi64(a, b);
}

__m256i test_mm256_slli_si256(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_si256
  // CHECK: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  return _mm256_slli_si256(a, 3);
}

__m128i test_mm_sllv_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sllv_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sllv_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_sllv_epi32((__m128i)(__v4si){1, -2, 3, -4}, (__m128i)(__v4si){1, 2, 3, -4}), 2, -8, 24, 0));

__m256i test_mm256_sllv_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sllv_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sllv_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_sllv_epi32((__m256i)(__v8si){1, -2, 3, -4, 5, -6, 7, -8}, (__m256i)(__v8si){1, 2, 3, 4, -17, 31, 33, 29}), 2, -8, 24, -64, 0, 0, 0, 0));

__m128i test_mm_sllv_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sllv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.psllv.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_sllv_epi64(a, b);
}
TEST_CONSTEXPR(match_m128i(_mm_sllv_epi64((__m128i)(__v2di){1, -3}, (__m128i)(__v2di){8, 63}), 256, 0x8000000000000000ULL));

__m256i test_mm256_sllv_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sllv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psllv.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_sllv_epi64(a, b);
}
TEST_CONSTEXPR(match_m256i(_mm256_sllv_epi64((__m256i)(__v4di){1, -2, 3, -4}, (__m256i)(__v4di){1, 2, 3, -4}), 2, -8, 24, 0));

__m256i test_mm256_sra_epi16(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_sra_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm256_sra_epi16(a, b);
}

__m256i test_mm256_sra_epi32(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_sra_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm256_sra_epi32(a, b);
}

__m256i test_mm256_srai_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_srai_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrai.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi16(a, 3);
}
TEST_CONSTEXPR(match_v16hi(_mm256_srai_epi16((__m256i)(__v16hi){-32768, 32767, -3, -2, -1, 0, 1, 2, -32768, 32767, -3, -2, -1, 0, 1, 2}, 1), -16384, 16383, -2, -1, -1, 0, 0, 1, -16384, 16383, -2, -1, -1, 0, 0, 1));

__m256i test_mm256_srai_epi16_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_srai_epi16_2
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrai.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi16(a, b);
}

__m256i test_mm256_srai_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_srai_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrai.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi32(a, 3);
}
TEST_CONSTEXPR(match_v8si(_mm256_srai_epi32((__m256i)(__v8si){-32768, 32767, -3, -2, -1, 0, 1, 2}, 1), -16384, 16383, -2, -1, -1, 0, 0, 1));

__m256i test_mm256_srai_epi32_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_srai_epi32_2
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrai.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi32(a, b);
}

__m128i test_mm_srav_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srav_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psrav.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_srav_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_srav_epi32((__m128i)(__v4si){1, -2, 3, -4}, (__m128i)(__v4si){1, 2, 3, -4}), 0, -1, 0, -1));

__m256i test_mm256_srav_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srav_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrav.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_srav_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_srav_epi32((__m256i)(__v8si){1, -2, 3, -4, 5, -6, 7, -8}, (__m256i)(__v8si){1, 2, 3, 4, -17, 31, 33, 29}), 0, -1, 0, -1, 0, -1, 0, -1));

__m256i test_mm256_srl_epi16(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm256_srl_epi16(a, b);
}

__m256i test_mm256_srl_epi32(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi32
  // CHECK:call <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm256_srl_epi32(a, b);
}

__m256i test_mm256_srl_epi64(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm256_srl_epi64(a, b);
}

__m256i test_mm256_srli_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi16(a, 3);
}
TEST_CONSTEXPR(match_v16hi(_mm256_srli_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7));

__m256i test_mm256_srli_epi16_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_srli_epi16_2
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi16(a, b);
}

__m256i test_mm256_srli_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi32(a, 3);
}
TEST_CONSTEXPR(match_v8si(_mm256_srli_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, 31), 0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0));

__m256i test_mm256_srli_epi32_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_srli_epi32_2
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi32(a, b);
}

__m256i test_mm256_srli_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psrli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi64(a, 3);
}
TEST_CONSTEXPR(match_v4di(_mm256_srli_epi64((__m256i)(__v4di){0, 1, 2, 3}, 33), 0, 0x0, 0x0, 0x0));

__m256i test_mm256_srli_epi64_2(__m256i a, int b) {
  // CHECK-LABEL: test_mm256_srli_epi64_2
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psrli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi64(a, b);
}

__m256i test_mm256_srli_si256(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_si256
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  return _mm256_srli_si256(a, 3);
}

__m128i test_mm_srlv_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srlv_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psrlv.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_srlv_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_srlv_epi32((__m128i)(__v4si){1, -2, 3, -4}, (__m128i)(__v4si){1, 2, 3, -4}), 0, 1073741823, 0, 0));

__m256i test_mm256_srlv_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srlv_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrlv.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_srlv_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_srlv_epi32((__m256i)(__v8si){1, -2, 3, -4, 5, -6, 7, -8}, (__m256i)(__v8si){1, 2, 3, 4, -17, 31, 33, 29}), 0, 1073741823, 0, 268435455, 0, 1, 0, 7));

__m128i test_mm_srlv_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srlv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx2.psrlv.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_srlv_epi64(a, b);
}
TEST_CONSTEXPR(match_m128i(_mm_srlv_epi64((__m128i)(__v2di){1, -3}, (__m128i)(__v2di){8, 63}), 0, 1));

__m256i test_mm256_srlv_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srlv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx2.psrlv.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_srlv_epi64(a, b);
}
TEST_CONSTEXPR(match_m256i(_mm256_srlv_epi64((__m256i)(__v4di){1, -2, 3, -4}, (__m256i)(__v4di){1, 2, 3, -4}), 0, 0x3FFFFFFFFFFFFFFFULL, 0, 0));

__m256i test_mm256_stream_load_si256(__m256i const *a) {
  // CHECK-LABEL: test_mm256_stream_load_si256
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 32, !nontemporal
  return _mm256_stream_load_si256(a);
}

__m256i test_mm256_stream_load_si256_void(const void *a) {
  // CHECK-LABEL: test_mm256_stream_load_si256_void
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 32, !nontemporal
  return _mm256_stream_load_si256(a);
}

__m256i test_mm256_sub_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi8
  // CHECK: sub <32 x i8>
  return _mm256_sub_epi8(a, b);
}

TEST_CONSTEXPR(match_v32qi(_mm256_sub_epi8((__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}),  -63, -63, 63, 71, 63, -63, -77, -79, -81, 83, 63, -63, 63, 63, -93, 63, -63, 63, 63, -103, 105, -107, 109, 111, -113, -115, -63, 119, -63, 63, 125, 63));

__m256i test_mm256_sub_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi16
  // CHECK: sub <16 x i16>
  return _mm256_sub_epi16(a, b);
}

TEST_CONSTEXPR(match_v16hi(_mm256_sub_epi16((__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}),  -31, -35, 31, -31, -41, 43, -45, -47, -49, -51, -31, 55, -31, -59, 61, 63));

__m256i test_mm256_sub_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi32
  // CHECK: sub <8 x i32>
  return _mm256_sub_epi32(a, b);
}

TEST_CONSTEXPR(match_v8si(_mm256_sub_epi32((__m256i)(__v8si){ 16, 17, 18, -19, 20, -21, 22, 23}, (__m256i)(__v8si){ -1, 2, 3, 4, 5, 6, 7, 8}),  17, 15, 15, -23, 15, -27, 15, 15));

__m256i test_mm256_sub_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi64
  // CHECK: sub <4 x i64>
  return _mm256_sub_epi64(a, b);
}

TEST_CONSTEXPR(match_v4di(_mm256_sub_epi64((__m256i)(__v4di){ 8, -9, 10, 11}, (__m256i)(__v4di){ -1, -2, 3, 4}),  9, -7, 7, 7));

__m256i test_mm256_subs_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epi8
  // CHECK: call <32 x i8> @llvm.ssub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_subs_epi8(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_subs_epi8((__m256i)(__v32qs){0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +100, +50, -100, +20, +80, -50, +120, -20, -100, -50, +100, -20, -80, +50, -120, +20}, (__m256i)(__v32qs){0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -50, -80, +50, -110, -60, +30, -20, +10, -50, -80, +50, -110, -60, +30, -20, +10}), 0, +2, +4, +6, +8, +10, +12, +14, +16, +18, +20, +22, +24, +26, +28, +30, +127, +127, -128, +127, +127, -80, +127, -30, -50, +30, +50, +90, -20, +20, -100, +10));

__m256i test_mm256_subs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epi16
  // CHECK: call <16 x i16> @llvm.ssub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_subs_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_subs_epi16((__m256i)(__v16hi){0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, +32000, -32000, +32000, -32000}, (__m256i)(__v16hi){0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, -800, +800, +800, -800}),0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, +32767, -32768, +31200, -31200));

__m256i test_mm256_subs_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epu8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_subs_epu8(a, b);
}
TEST_CONSTEXPR(match_v32qu(_mm256_subs_epu8((__m256i)(__v32qu){0, 0, 0, 0, +64, +64, +64, +64, +64, +64, +127, +127, +127, +127, +127, +127, +128, +128, +128, +128, +128, +128, +192, +192, +192, +192, +192, +192, +255, +255, +255, +255}, (__m256i)(__v32qu){0, +127, +128, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +64, +127, +128, +192, +255, 0, +127, +128, +255}), 0, 0, 0, 0, +64, 0, 0, 0, 0, 0, +127, +63, 0, 0, 0, 0, +128, +64, +1, 0, 0, 0, +192, +128, +65, +64, 0, 0, +255, +128, +127, 0));

__m256i test_mm256_subs_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epu16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_subs_epu16(a, b);
}
TEST_CONSTEXPR(match_v16hu(_mm256_subs_epu16((__m256i)(__v16hu){0, 0, 0, 0, +32767, +32767, +32767, +32767, +32768, +32768, +32768, +32768, +65535, +65535, +65535, +65535}, (__m256i)(__v16hu){0, +32767, +32768, +65535, 0, +32767, +32768, +65535, 0, +32767, +32768, +65535, 0, +32767, +32768, +65535}), 0, 0, 0, 0, +32767, 0, 0, 0, +32768, +1, 0, 0, +65535, +32768, +32767, 0));

__m256i test_mm256_unpackhi_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  return _mm256_unpackhi_epi8(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_unpackhi_epi8((__m256i)(__v32qi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, (__m256i)(__v32qi){32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}), 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));

__m256i test_mm256_unpackhi_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  return _mm256_unpackhi_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_unpackhi_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, (__m256i)(__v16hi){16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31));

__m256i test_mm256_unpackhi_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  return _mm256_unpackhi_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_unpackhi_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, (__m256i)(__v8si){8, 9, 10, 11, 12, 13, 14, 15}), 2, 10, 3, 11, 6, 14, 7, 15));

__m256i test_mm256_unpackhi_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  return _mm256_unpackhi_epi64(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_unpackhi_epi64((__m256i)(__v4di){0, 1, 2, 3}, (__m256i)(__v4di){ 4, 5, 6, 7}), 1, 5, 3, 7));

__m256i test_mm256_unpacklo_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  return _mm256_unpacklo_epi8(a, b);
}
TEST_CONSTEXPR(match_v32qi(_mm256_unpacklo_epi8((__m256i)(__v32qi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, (__m256i)(__v32qi){32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}), 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55));

__m256i test_mm256_unpacklo_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  return _mm256_unpacklo_epi16(a, b);
}
TEST_CONSTEXPR(match_v16hi(_mm256_unpacklo_epi16((__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, (__m256i)(__v16hi){16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27));

__m256i test_mm256_unpacklo_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  return _mm256_unpacklo_epi32(a, b);
}
TEST_CONSTEXPR(match_v8si(_mm256_unpacklo_epi32((__m256i)(__v8si){0, 1, 2, 3, 4, 5, 6, 7}, (__m256i)(__v8si){ 8, 9, 10, 11, 12, 13, 14, 15}), 0, 8, 1, 9, 4, 12, 5, 13));

__m256i test_mm256_unpacklo_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_unpacklo_epi64(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_unpacklo_epi64((__m256i)(__v4di){0, 1, 2, 3}, (__m256i)(__v4di){ 4, 5, 6, 7}), 0, 4, 2, 6));

__m256i test_mm256_xor_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_xor_si256
  // CHECK: xor <4 x i64>
  return _mm256_xor_si256(a, b);
}
TEST_CONSTEXPR(match_v4di(_mm256_xor_si256((__m256i)(__v4di){0, -1, 0, -1}, (__m256i)(__v4di){0, 0, -1, -1}), 0, -1, -1, 0));

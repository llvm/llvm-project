// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK


#include <immintrin.h>
#include "builtin_test_helpers.h"

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse41-intrinsics-fast-isel.ll

__m128i test_mm_blend_epi16(__m128i V1, __m128i V2) {
  // CHECK-LABEL: test_mm_blend_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}
TEST_CONSTEXPR(match_v8hi(_mm_blend_epi16(((__m128i)(__v8hi){1,2,3,4,5,6,7,8}),((__m128i)(__v8hi){-1,-2,-3,-4,-5,-6,-7,-8}),0x00),1,2,3,4,5,6,7,8));
TEST_CONSTEXPR(match_v8hi(_mm_blend_epi16(((__m128i)(__v8hi){1,2,3,4,5,6,7,8}),((__m128i)(__v8hi){-1,-2,-3,-4,-5,-6,-7,-8}),0x5A),1,-2,3,-4,-5,6,-7,8));
TEST_CONSTEXPR(match_v8hi(_mm_blend_epi16(((__m128i)(__v8hi){1,2,3,4,5,6,7,8}),((__m128i)(__v8hi){-1,-2,-3,-4,-5,-6,-7,-8}),0x94),1,2,-3,4,-5,6,7,-8));
TEST_CONSTEXPR(match_v8hi(_mm_blend_epi16(((__m128i)(__v8hi){1,2,3,4,5,6,7,8}),((__m128i)(__v8hi){-1,-2,-3,-4,-5,-6,-7,-8}),0xFF),-1,-2,-3,-4,-5,-6,-7,-8));

__m128d test_mm_blend_pd(__m128d V1, __m128d V2) {
  // CHECK-LABEL: test_mm_blend_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_blend_pd(V1, V2, 2);
}
TEST_CONSTEXPR(match_m128d(_mm_blend_pd(((__m128d){1.0, 2.0}), ((__m128d){3.0, 4.0}), 0), 1.0, 2.0));
TEST_CONSTEXPR(match_m128d(_mm_blend_pd(((__m128d){1.0, 2.0}), ((__m128d){3.0, 4.0}), 1), 3.0, 2.0));
TEST_CONSTEXPR(match_m128d(_mm_blend_pd(((__m128d){1.0, 2.0}), ((__m128d){3.0, 4.0}), 2), 1.0, 4.0));
TEST_CONSTEXPR(match_m128d(_mm_blend_pd(((__m128d){1.0, 2.0}), ((__m128d){3.0, 4.0}), 3), 3.0, 4.0));

__m128 test_mm_blend_ps(__m128 V1, __m128 V2) {
  // CHECK-LABEL: test_mm_blend_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 6);
}
TEST_CONSTEXPR(match_m128(_mm_blend_ps(((__m128){1.0f, 2.0f, 3.0f, 4.0f}), ((__m128){5.0f, 6.0f, 7.0f, 8.0f}), 0x0), 1.0f, 2.0f, 3.0f, 4.0f));
TEST_CONSTEXPR(match_m128(_mm_blend_ps(((__m128){1.0f, 2.0f, 3.0f, 4.0f}), ((__m128){5.0f, 6.0f, 7.0f, 8.0f}), 0x5), 5.0f, 2.0f, 7.0f, 4.0f));
TEST_CONSTEXPR(match_m128(_mm_blend_ps(((__m128){1.0f, 2.0f, 3.0f, 4.0f}), ((__m128){5.0f, 6.0f, 7.0f, 8.0f}), 0xA), 1.0f, 6.0f, 3.0f, 8.0f));
TEST_CONSTEXPR(match_m128(_mm_blend_ps(((__m128){1.0f, 2.0f, 3.0f, 4.0f}), ((__m128){5.0f, 6.0f, 7.0f, 8.0f}), 0xF), 5.0f, 6.0f, 7.0f, 8.0f));

__m128i test_mm_blendv_epi8(__m128i V1, __m128i V2, __m128i V3) {
  // CHECK-LABEL: test_mm_blendv_epi8
  // CHECK: call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_blendv_epi8(V1, V2, V3);
}
TEST_CONSTEXPR(match_v16qi(_mm_blendv_epi8((__m128i)(__v16qs){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},(__m128i)(__v16qs){-99,-98,97,-96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84},(__m128i)(__v16qs){-1,-1,0,-1,0,0,0,0,0,-1,-1,-1,0,0,-1,0}), -99, -98, 2, -96, 4, 5, 6, 7, 8, -90, -89, -88, 12, 13, -85, 15));

__m128d test_mm_blendv_pd(__m128d V1, __m128d V2, __m128d V3) {
  // CHECK-LABEL: test_mm_blendv_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_blendv_pd(V1, V2, V3);
}
TEST_CONSTEXPR(match_m128d(_mm_blendv_pd((__m128d)(__v2df){2.0, -4.0},(__m128d)(__v2df){-111.0, +222.0},(__m128d)(__v2df){2.0, -2.0}), 2.0, 222.0));

__m128 test_mm_blendv_ps(__m128 V1, __m128 V2, __m128 V3) {
  // CHECK-LABEL: test_mm_blendv_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.blendvps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_blendv_ps(V1, V2, V3);
}
TEST_CONSTEXPR(match_m128(_mm_blendv_ps((__m128)(__v4sf){0.0f, 1.0f, 2.0f, 3.0f},(__m128)(__v4sf){-100.0f, -101.0f, -102.0f, -103.0f},(__m128)(__v4sf){-1.0f, 2.0f, -3.0f, 0.0f}), -100.0f, 1.0f, -102.0f, 3.0f));

__m128d test_mm_ceil_pd(__m128d x) {
  // CHECK-LABEL: test_mm_ceil_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 2)
  return _mm_ceil_pd(x);
}

__m128 test_mm_ceil_ps(__m128 x) {
  // CHECK-LABEL: test_mm_ceil_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ps(x);
}

__m128d test_mm_ceil_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_ceil_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2)
  return _mm_ceil_sd(x, y);
}

__m128 test_mm_ceil_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_ceil_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ss(x, y);
}

__m128i test_mm_cmpeq_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi64
  // CHECK: icmp eq <2 x i64>
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_cmpeq_epi64(A, B);
}
TEST_CONSTEXPR(match_v2di(_mm_cmpeq_epi64((__m128i)(__v2di){+1, -8}, (__m128i)(__v2di){-10, -8}), 0, -1));

__m128i test_mm_cvtepi8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi16
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: sext <8 x i8> {{.*}} to <8 x i16>
  return _mm_cvtepi8_epi16(a);
}

TEST_CONSTEXPR(match_v8hi(_mm_cvtepi8_epi16(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), -3, 2, -1, 0, 1, -2, 3, -4));

__m128i test_mm_cvtepi8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi32
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i8> {{.*}} to <4 x i32>
  return _mm_cvtepi8_epi32(a);
}

TEST_CONSTEXPR(match_v4si(_mm_cvtepi8_epi32(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), -3, 2, -1, 0));

__m128i test_mm_cvtepi8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi64
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i8> {{.*}} to <2 x i64>
  return _mm_cvtepi8_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepi8_epi64(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), -3, 2));

__m128i test_mm_cvtepi16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi32
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i16> {{.*}} to <4 x i32>
  return _mm_cvtepi16_epi32(a);
}

TEST_CONSTEXPR(match_v4si(_mm_cvtepi16_epi32(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), -300, 2, -1, 0));

__m128i test_mm_cvtepi16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi64
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i16> {{.*}} to <2 x i64>
  return _mm_cvtepi16_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepi16_epi64(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), -300, 2));

__m128i test_mm_cvtepi32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi32_epi64
  // CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i32> {{.*}} to <2 x i64>
  return _mm_cvtepi32_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepi32_epi64(_mm_setr_epi32(-70000, 2, -1, 0)), -70000, 2));

__m128i test_mm_cvtepu8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi16
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext <8 x i8> {{.*}} to <8 x i16>
  return _mm_cvtepu8_epi16(a);
}

TEST_CONSTEXPR(match_v8hi(_mm_cvtepu8_epi16(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), 253, 2, 255, 0, 1, 254, 3, 252));

__m128i test_mm_cvtepu8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi32
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i8> {{.*}} to <4 x i32>
  return _mm_cvtepu8_epi32(a);
}

TEST_CONSTEXPR(match_v4si(_mm_cvtepu8_epi32(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), 253, 2, 255, 0));

__m128i test_mm_cvtepu8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi64
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i8> {{.*}} to <2 x i64>
  return _mm_cvtepu8_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepu8_epi64(_mm_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0)), 253, 2));

__m128i test_mm_cvtepu16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi32
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i16> {{.*}} to <4 x i32>
  return _mm_cvtepu16_epi32(a);
}

TEST_CONSTEXPR(match_v4si(_mm_cvtepu16_epi32(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), 65236, 2, 65535, 0));

__m128i test_mm_cvtepu16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi64
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i16> {{.*}} to <2 x i64>
  return _mm_cvtepu16_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepu16_epi64(_mm_setr_epi16(-300, 2, -1, 0, 1, -2, 3, -4)), 65236, 2));

__m128i test_mm_cvtepu32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu32_epi64
  // CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i32> {{.*}} to <2 x i64>
  return _mm_cvtepu32_epi64(a);
}

TEST_CONSTEXPR(match_v2di(_mm_cvtepu32_epi64(_mm_setr_epi32(-70000, 2, -1, 0)), 4294897296, 2));

__m128d test_mm_dp_pd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_dp_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.dppd(<2 x double> {{.*}}, <2 x double> {{.*}}, i8 7)
  return _mm_dp_pd(x, y, 7);
}

__m128 test_mm_dp_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_dp_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.dpps(<4 x float> {{.*}}, <4 x float> {{.*}}, i8 7)
  return _mm_dp_ps(x, y, 7);
}

int test_mm_extract_epi8(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi8
  // CHECK: extractelement <16 x i8> %{{.*}}, {{i32|i64}} 1
  // CHECK: zext i8 %{{.*}} to i32
  return _mm_extract_epi8(x, 1);
}
TEST_CONSTEXPR(_mm_extract_epi8(((__m128i)(__v16qi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 20) == 4);

int test_mm_extract_epi32(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi32
  // CHECK: extractelement <4 x i32> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi32(x, 1);
}
TEST_CONSTEXPR(_mm_extract_epi32(((__m128i)(__v4si){1, 3, 5, 7}), 10) == 5);

long long test_mm_extract_epi64(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi64
  // CHECK: extractelement <2 x i64> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi64(x, 1);
}
TEST_CONSTEXPR(_mm_extract_epi64(((__m128i)(__v2di){11, 22}), 5) == 22);

int test_mm_extract_ps(__m128 x) {
  // CHECK-LABEL: test_mm_extract_ps
  // CHECK: extractelement <4 x float> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_ps(x, 1);
}
TEST_CONSTEXPR(_mm_extract_ps(((__m128){1.25f, 2.5f, 3.75f, 5.0f}), 6) == __builtin_bit_cast(int, 3.75f));

__m128d test_mm_floor_pd(__m128d x) {
  // CHECK-LABEL: test_mm_floor_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 1)
  return _mm_floor_pd(x);
}

__m128 test_mm_floor_ps(__m128 x) {
  // CHECK-LABEL: test_mm_floor_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 1)
  return _mm_floor_ps(x);
}

__m128d test_mm_floor_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_floor_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1)
  return _mm_floor_sd(x, y);
}

__m128 test_mm_floor_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_floor_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
  return _mm_floor_ss(x, y);
}

__m128i test_mm_insert_epi8(__m128i x, char b) {
  // CHECK-LABEL: test_mm_insert_epi8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi8(x, b, 1);
}
TEST_CONSTEXPR(match_v16qi(_mm_insert_epi8(((__m128i)(__v16qi){ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 101, 33), 0, 101, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15));

__m128i test_mm_insert_epi32(__m128i x, int b) {
  // CHECK-LABEL: test_mm_insert_epi32
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi32(x, b, 1);
}
TEST_CONSTEXPR(match_v4si(_mm_insert_epi32(((__m128i)(__v4si){0, 1, 2, 3}), 5678, 18), 0, 1, 5678, 3));

#ifdef __x86_64__
__m128i test_mm_insert_epi64(__m128i x, long long b) {
  // X64-LABEL: test_mm_insert_epi64
  // X64: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi64(x, b, 1);
}
TEST_CONSTEXPR(match_v2di(_mm_insert_epi64(((__m128i)(__v2di){100, 200}), -999, 9), 100, -999));
#endif

__m128 test_mm_insert_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_insert_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.insertps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_insert_ps(x, y, 4);
}

__m128i test_mm_max_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi8
  // CHECK: call <16 x i8> @llvm.smax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_max_epi8(x, y);
}

TEST_CONSTEXPR(match_v16qi(_mm_max_epi8((__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16));

__m128i test_mm_max_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi32
  // CHECK: call <4 x i32> @llvm.smax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_max_epi32(x, y);
}

TEST_CONSTEXPR(match_v4si(_mm_max_epi32((__m128i)(__v4si){-1, +2, -3, +4}, (__m128i)(__v4si){+1, -2, +3, -4}), +1, +2, +3, +4 ));

__m128i test_mm_max_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu16
  // CHECK: call <8 x i16> @llvm.umax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_max_epu16(x, y);
}

TEST_CONSTEXPR(match_v8hu(_mm_max_epu16((__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 3, 4, 5, 7, 9, 11, 13, 15));

__m128i test_mm_max_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu32
  // CHECK: call <4 x i32> @llvm.umax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_max_epu32(x, y);
}

TEST_CONSTEXPR(match_v4su(_mm_max_epu32((__m128i)(__v4su){1, 3, 5, 7}, (__m128i)(__v4su){3, 4, 5, 6}), 3, 4, 5, 7));

__m128i test_mm_min_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi8
  // CHECK: call <16 x i8> @llvm.smin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_min_epi8(x, y);
}

TEST_CONSTEXPR(match_v16qi(_mm_min_epi8((__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16));

__m128i test_mm_min_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi32
  // CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_min_epi32(x, y);
}

TEST_CONSTEXPR(match_v4si(_mm_min_epi32((__m128i)(__v4si){-1, +2, -3, +4}, (__m128i)(__v4si){+1, -2, +3, -4}), -1, -2, -3, -4 ));

__m128i test_mm_min_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu16
  // CHECK: call <8 x i16> @llvm.umin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_min_epu16(x, y);
}

TEST_CONSTEXPR(match_v8hu(_mm_min_epu16((__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 1, 3, 5, 6, 7, 8, 9, 10));

__m128i test_mm_min_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu32
  // CHECK: call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_min_epu32(x, y);
}

TEST_CONSTEXPR(match_v4su(_mm_min_epu32((__m128i)(__v4su){1, 3, 5, 7}, (__m128i)(__v4su){3, 4, 5, 6}), 1, 3, 5, 6));

__m128i test_mm_minpos_epu16(__m128i x) {
  // CHECK-LABEL: test_mm_minpos_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16> %{{.*}})
  return _mm_minpos_epu16(x);
}

__m128i test_mm_mpsadbw_epu8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mpsadbw_epu8
  // CHECK: call <8 x i16> @llvm.x86.sse41.mpsadbw(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 1)
  return _mm_mpsadbw_epu8(x, y, 1);
}

__m128i test_mm_mul_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mul_epi32
  // CHECK: shl <2 x i64> %{{.*}}, splat (i64 32)
  // CHECK: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // CHECK: shl <2 x i64> %{{.*}}, splat (i64 32)
  // CHECK: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mul_epi32(x, y);
}
TEST_CONSTEXPR(match_m128i(_mm_mul_epi32((__m128i)(__v4si){+1, -2, +3, -4}, (__m128i)(__v4si){-16, -14, +12, +10}), -16, 36));

__m128i test_mm_mullo_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mullo_epi32
  // CHECK: mul <4 x i32>
  return _mm_mullo_epi32(x, y);
}
TEST_CONSTEXPR(match_v4si(_mm_mullo_epi32((__m128i)(__v4si){+1, -2, +3, -4}, (__m128i)(__v4si){-16, +14, +12, -10}), -16, -28, +36, +40));

__m128i test_mm_packus_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_packus_epi32
  // CHECK: call <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_packus_epi32(x, y);
}
TEST_CONSTEXPR(match_v8hi(_mm_packus_epi32((__m128i)(__v4si){40000, -50000, 32767, -32768}, (__m128i)(__v4si){0, 1, -1, 70000}), -25536, 0, 32767, 0, 0, 1, 0, -1));

__m128d test_mm_round_pd(__m128d x) {
  // CHECK-LABEL: test_mm_round_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 4)
  return _mm_round_pd(x, 4);
}

__m128 test_mm_round_ps(__m128 x) {
  // CHECK-LABEL: test_mm_round_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 4)
  return _mm_round_ps(x, 4);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_round_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 4)
  return _mm_round_sd(x, y, 4);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_round_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 4)
  return _mm_round_ss(x, y, 4);
}

__m128i test_mm_stream_load_si128(__m128i const *a) {
  // CHECK-LABEL: test_mm_stream_load_si128
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 16, !nontemporal
  return _mm_stream_load_si128(a);
}

__m128i test_mm_stream_load_si128_void(const void *a) {
  // CHECK-LABEL: test_mm_stream_load_si128_void
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 16, !nontemporal
  return _mm_stream_load_si128(a);
}

int test_mm_test_all_ones(__m128i x) {
  // CHECK-LABEL: test_mm_test_all_ones
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_all_ones(x);
}
TEST_CONSTEXPR(_mm_test_all_ones(((__m128i)(__v2di){-1, -1})) == 1);
TEST_CONSTEXPR(_mm_test_all_ones(((__m128i)(__v2di){-1,  0})) == 0);
TEST_CONSTEXPR(_mm_test_all_ones(((__m128i)(__v4si){-1, -1, -1, 0x7FFFFFFF})) == 0);

int test_mm_test_all_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_all_zeros
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestz(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_all_zeros(x, y);
}
TEST_CONSTEXPR(_mm_test_all_zeros(((__m128i)(__v2di){0,0}), ((__m128i)(__v2di){0,0})) == 1);
TEST_CONSTEXPR(_mm_test_all_zeros(((__m128i)(__v2di){0xFF00,0}), ((__m128i)(__v2di){0x00FF,0})) == 1);
TEST_CONSTEXPR(_mm_test_all_zeros(((__m128i)(__v2di){1,0}), ((__m128i)(__v2di){-1,0})) == 0);
TEST_CONSTEXPR(_mm_test_all_zeros(((__m128i)(__v2di){0,1}), ((__m128i)(__v2di){0,-1})) == 0);

int test_mm_test_mix_ones_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_mix_ones_zeros
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_mix_ones_zeros(x, y);
}
TEST_CONSTEXPR(_mm_test_mix_ones_zeros(((__m128i)(__v2di){0xFF, 0}), ((__m128i)(__v2di){0xF0, 1})) == 1);
TEST_CONSTEXPR(_mm_test_mix_ones_zeros(((__m128i)(__v2di){0xF0, 0}), ((__m128i)(__v2di){0x0F, 0})) == 0);
TEST_CONSTEXPR(_mm_test_mix_ones_zeros(((__m128i)(__v2di){-1, -1}), ((__m128i)(__v2di){1, 0})) == 0);
TEST_CONSTEXPR(_mm_test_mix_ones_zeros(((__m128i)(__v2di){0, 0}), ((__m128i)(__v2di){0, 0})) == 0);

int test_mm_testc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testc_si128
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testc_si128(x, y);
}
TEST_CONSTEXPR(_mm_testc_si128((__m128i)(__v2di){0,0}, (__m128i)(__v2di){0,0}) == 1);
TEST_CONSTEXPR(_mm_testc_si128((__m128i)(__v2di){1,0}, (__m128i)(__v2di){-1,0}) == 0);
TEST_CONSTEXPR(_mm_testc_si128((__m128i)(__v2di){0,-1}, (__m128i)(__v2di){0,1}) == 1);

int test_mm_testnzc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testnzc_si128
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testnzc_si128(x, y);
}
TEST_CONSTEXPR(_mm_testnzc_si128((__m128i)(__v2di){3,0}, (__m128i)(__v2di){1,1}) == 1);
TEST_CONSTEXPR(_mm_testnzc_si128((__m128i)(__v2di){32,-1}, (__m128i)(__v2di){15,0}) == 0);
TEST_CONSTEXPR(_mm_testnzc_si128((__m128i)(__v2di){0,999}, (__m128i)(__v2di){0,999}) == 0);

int test_mm_testz_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testz_si128
  // CHECK: call {{.*}}i32 @llvm.x86.sse41.ptestz(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testz_si128(x, y);
}
TEST_CONSTEXPR(_mm_testz_si128((__m128i)(__v2di){0,0}, (__m128i)(__v2di){0,0}) == 1);
TEST_CONSTEXPR(_mm_testz_si128((__m128i)(__v2di){1,0}, (__m128i)(__v2di){-1,0}) == 0);
TEST_CONSTEXPR(_mm_testz_si128((__m128i)(__v2di){1,0}, (__m128i)(__v2di){0,1}) == 1);

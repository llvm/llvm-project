// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"


__m128i test_mm_madd52hi_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52hi_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52hi_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_epu64(
                              (__m128i)((__v2du){50, 100}),
                              (__m128i)((__v2du){10, 20}),
                              (__m128i)((__v2du){5, 6})),
                          50, 100));

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0})),
                          0xFFFFFFFFFFFFEull, 0));

__m256i test_mm256_madd52hi_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52hi_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52hi_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_epu64(
                              (__m256i)((__v4du){100, 200, 300, 400}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){5, 6, 7, 8})),
                          100, 200, 300, 400));

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull,
                                                 0xFFFFFFFFFFFFFull, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull,
                                                 0xFFFFFFFFFFFFFull, 0, 0})),
                          0xFFFFFFFFFFFFEull, 0xFFFFFFFFFFFFEull, 0, 0));

__m128i test_mm_madd52lo_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52lo_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52lo_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){10, 0}),
                              (__m128i)((__v2du){5, 0})),
                          50, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){1, 2}),
                              (__m128i)((__v2du){10, 20}),
                              (__m128i)((__v2du){2, 3})),
                          21, 62));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0}),
                              (__m128i)((__v2du){1, 0})),
                          0xFFFFFFFFFFFFFull, 0));

__m256i test_mm256_madd52lo_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52lo_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52lo_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_epu64(
                              (__m256i)((__v4du){1, 2, 3, 4}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){2, 3, 4, 5})),
                          21, 62, 123, 204));

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){1, 0, 0, 0})),
                          0xFFFFFFFFFFFFFull, 0, 0, 0));

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0x1F000000000000ull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){2, 0, 0, 0})),
                          0xE000000000000ull, 0, 0, 0));

__m128i test_mm_madd52hi_avx_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_madd52hi_avx_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52hi_avx_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_avx_epu64(
                              (__m128i)((__v2du){50, 100}),
                              (__m128i)((__v2du){10, 20}),
                              (__m128i)((__v2du){5, 6})),
                          50, 100));

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_avx_epu64(
                              (__m128i)((__v2du){100, 0}),
                              (__m128i)((__v2du){10, 0}),
                              (__m128i)((__v2du){5, 0})),
                          100, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_avx_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0})),
                          0xFFFFFFFFFFFFEull, 0));
                        
__m256i test_mm256_madd52hi_avx_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_madd52hi_avx_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52hi_avx_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_avx_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull,
                                                 0xFFFFFFFFFFFFFull, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull,
                                                 0xFFFFFFFFFFFFFull, 0, 0})),
                          0xFFFFFFFFFFFFEull, 0xFFFFFFFFFFFFEull, 0, 0));

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_avx_epu64(
                              (__m256i)((__v4du){100, 200, 300, 400}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){5, 6, 7, 8})),
                          100, 200, 300, 400));

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_avx_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0})),
                          0xFFFFFFFFFFFFEull, 0, 0, 0));

__m128i test_mm_madd52lo_avx_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_madd52lo_avx_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52lo_avx_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_avx_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){10, 0}),
                              (__m128i)((__v2du){5, 0})),
                          50, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_avx_epu64(
                              (__m128i)((__v2du){100, 0}),
                              (__m128i)((__v2du){20, 0}),
                              (__m128i)((__v2du){30, 0})),
                          700, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_avx_epu64(
                              (__m128i)((__v2du){1, 2}),
                              (__m128i)((__v2du){10, 20}),
                              (__m128i)((__v2du){2, 3})),
                          21, 62));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_avx_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0}),
                              (__m128i)((__v2du){1, 0})),
                          0xFFFFFFFFFFFFFull, 0));

__m256i test_mm256_madd52lo_avx_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_madd52lo_avx_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52lo_avx_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_avx_epu64(
                              (__m256i)((__v4du){1, 2, 3, 4}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){2, 3, 4, 5})),
                          21, 62, 123, 204));



TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_avx_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){1, 0, 0, 0})),
                          0xFFFFFFFFFFFFFull, 0, 0, 0));

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_avx_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0x1F000000000000ull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){2, 0, 0, 0})),
                          0xE000000000000ull, 0, 0, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_avx_epu64(
                              (__m128i)((__v2du){5, 10}),
                              (__m128i)((__v2du){100, 200}),
                              (__m128i)((__v2du){7, 8})),
                          705, 1610));

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_avx_epu64(
                              (__m256i)((__v4du){1, 2, 3, 4}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){2, 3, 4, 5})),
                          21, 62, 123, 204));


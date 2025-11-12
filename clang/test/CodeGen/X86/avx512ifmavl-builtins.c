// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=i386-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=i386-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=i386-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=i386-apple-darwin -target-feature +avx512ifma -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_madd52hi_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_madd52hi_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52hi_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_epu64(
                              (__m128i)((__v2du){100, 0}),
                              (__m128i)((__v2du){10, 0}),
                              (__m128i)((__v2du){5, 0})),
                          100, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52hi_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0}),
                              (__m128i)((__v2du){0xFFFFFFFFFFFFFull, 0})),
                          0xFFFFFFFFFFFFEull, 0));

__m128i test_mm_mask_madd52hi_epu64(__m128i __W, __mmask8 __M, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_mask_madd52hi_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_madd52hi_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v2di(_mm_mask_madd52hi_epu64((__m128i)((__v2du){111, 222}),
                                                   0x0,
                                                   (__m128i)((__v2du){1, 2}),
                                                   (__m128i)((__v2du){10, 20})),
                          111, 222));

TEST_CONSTEXPR(match_v2di(_mm_mask_madd52hi_epu64((__m128i)((__v2du){10, 20}),
                                                   0x2,
                                                   (__m128i)((__v2du){0x1000000000000ULL, 0x1000000000000ULL}),
                                                   (__m128i)((__v2du){0x1000000000000ULL, 0x1000000000000ULL})),
                          10, 0x100000000014ULL));

__m128i test_mm_maskz_madd52hi_epu64(__mmask8 __M, __m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_maskz_madd52hi_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_madd52hi_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_maskz_madd52hi_epu64(0x3,
                                                    (__m128i)((__v2du){1, 2}),
                                                    (__m128i)((__v2du){10, 20}),
                                                    (__m128i)((__v2du){100, 200})),
                          1, 2));

TEST_CONSTEXPR(match_v2di(_mm_maskz_madd52hi_epu64(0x1,
                                                    (__m128i)((__v2du){0x1000000000000ULL, 0x1000000000000ULL}),
                                                    (__m128i)((__v2du){0x1000000000000ULL, 0x1000000000000ULL}),
                                                    (__m128i)((__v2du){0, 0})),
                          0x1000000000000ULL, 0));

__m256i test_mm256_madd52hi_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_madd52hi_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52hi_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_epu64(
                              (__m256i)((__v4du){100, 200, 300, 400}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){5, 6, 7, 8})),
                          100, 200, 300, 400));

TEST_CONSTEXPR(match_v4di(_mm256_madd52hi_epu64(
                              (__m256i)((__v4du){0, 0, 0, 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0}),
                              (__m256i)((__v4du){0xFFFFFFFFFFFFFull, 0, 0,
                                                 0})),
                          0xFFFFFFFFFFFFEull, 0, 0, 0));

__m256i test_mm256_mask_madd52hi_epu64(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_madd52hi_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_madd52hi_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v4di(_mm256_mask_madd52hi_epu64((__m256i)((__v4du){111, 222, 333, 444}),
                                                      0x0,
                                                      (__m256i)((__v4du){1, 2, 3, 4}),
                                                      (__m256i)((__v4du){10, 20, 30, 40})),
                          111, 222, 333, 444));

TEST_CONSTEXPR(match_v4di(_mm256_mask_madd52hi_epu64((__m256i)((__v4du){10, 20, 30, 40}),
                                                      0xA,
                                                      (__m256i)((__v4du){0x1000000000000ULL, 0x1000000000000ULL,
                                                                         0x1000000000000ULL, 0x1000000000000ULL}),
                                                      (__m256i)((__v4du){0x1000000000000ULL, 0x1000000000000ULL,
                                                                         0x1000000000000ULL, 0x1000000000000ULL})),
                          10, 0x100000000014ULL, 30, 0x100000000028ULL));

__m256i test_mm256_maskz_madd52hi_epu64(__mmask8 __M, __m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_maskz_madd52hi_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_madd52hi_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_maskz_madd52hi_epu64(0xF,
                                                       (__m256i)((__v4du){1, 2, 3, 4}),
                                                       (__m256i)((__v4du){10, 20, 30, 40}),
                                                       (__m256i)((__v4du){100, 200, 300, 400})),
                          1, 2, 3, 4));

TEST_CONSTEXPR(match_v4di(_mm256_maskz_madd52hi_epu64(0x5,
                                                       (__m256i)((__v4du){0x1000000000000ULL, 0x1000000000000ULL,
                                                                          0x1000000000000ULL, 0x1000000000000ULL}),
                                                       (__m256i)((__v4du){0x1000000000000ULL, 0x1000000000000ULL,
                                                                          0x1000000000000ULL, 0x1000000000000ULL}),
                                                       (__m256i)((__v4du){0, 0, 0, 0})),
                          0x1000000000000ULL, 0, 0x1000000000000ULL, 0));

__m128i test_mm_madd52lo_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_madd52lo_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52lo_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){0, 0}),
                              (__m128i)((__v2du){10, 0}),
                              (__m128i)((__v2du){5, 0})),
                          50, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){100, 0}),
                              (__m128i)((__v2du){20, 0}),
                              (__m128i)((__v2du){30, 0})),
                          700, 0));

TEST_CONSTEXPR(match_v2di(_mm_madd52lo_epu64(
                              (__m128i)((__v2du){1, 2}),
                              (__m128i)((__v2du){10, 20}),
                              (__m128i)((__v2du){2, 3})),
                          21, 62));

__m128i test_mm_mask_madd52lo_epu64(__m128i __W, __mmask8 __M, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_mask_madd52lo_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_madd52lo_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v2di(_mm_mask_madd52lo_epu64((__m128i)((__v2du){1000, 2000}),
                                                   0x3,
                                                   (__m128i)((__v2du){100, 200}),
                                                   (__m128i)((__v2du){20, 30})),
                          3000, 8000));

TEST_CONSTEXPR(match_v2di(_mm_mask_madd52lo_epu64((__m128i)((__v2du){111, 222}),
                                                   0x0,
                                                   (__m128i)((__v2du){1, 2}),
                                                   (__m128i)((__v2du){10, 20})),
                          111, 222));

__m128i test_mm_maskz_madd52lo_epu64(__mmask8 __M, __m128i __X, __m128i __Y, __m128i __Z) {
  // CHECK-LABEL: test_mm_maskz_madd52lo_epu64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_madd52lo_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v2di(_mm_maskz_madd52lo_epu64(0x3,
                                                    (__m128i)((__v2du){100, 200}),
                                                    (__m128i)((__v2du){20, 30}),
                                                    (__m128i)((__v2du){30, 40})),
                          700, 1400));

TEST_CONSTEXPR(match_v2di(_mm_maskz_madd52lo_epu64(0x1,
                                                    (__m128i)((__v2du){100, 0}),
                                                    (__m128i)((__v2du){20, 0}),
                                                    (__m128i)((__v2du){30, 0})),
                          700, 0));

__m256i test_mm256_madd52lo_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_madd52lo_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52lo_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_madd52lo_epu64(
                              (__m256i)((__v4du){1, 2, 3, 4}),
                              (__m256i)((__v4du){10, 20, 30, 40}),
                              (__m256i)((__v4du){2, 3, 4, 5})),
                          21, 62, 123, 204));

__m256i test_mm256_mask_madd52lo_epu64(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_madd52lo_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_madd52lo_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v4di(_mm256_mask_madd52lo_epu64((__m256i)((__v4du){1000, 2000, 3000, 4000}),
                                                      0xF,
                                                      (__m256i)((__v4du){100, 200, 300, 400}),
                                                      (__m256i)((__v4du){20, 30, 40, 50})),
                          3000, 8000, 15000, 24000));

TEST_CONSTEXPR(match_v4di(_mm256_mask_madd52lo_epu64((__m256i)((__v4du){111, 222, 333, 444}),
                                                      0x0,
                                                      (__m256i)((__v4du){1, 2, 3, 4}),
                                                      (__m256i)((__v4du){10, 20, 30, 40})),
                          111, 222, 333, 444));

TEST_CONSTEXPR(match_v4di(_mm256_mask_madd52lo_epu64((__m256i)((__v4du){11, 22, 33, 44}),
                                                      0x5,
                                                      (__m256i)((__v4du){100, 200, 300, 400}),
                                                      (__m256i)((__v4du){10, 20, 30, 40})),
                          1011, 22, 9033, 44));

__m256i test_mm256_maskz_madd52lo_epu64(__mmask8 __M, __m256i __X, __m256i __Y, __m256i __Z) {
  // CHECK-LABEL: test_mm256_maskz_madd52lo_epu64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_madd52lo_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v4di(_mm256_maskz_madd52lo_epu64(0xF,
                                                       (__m256i)((__v4du){100, 200, 300, 400}),
                                                       (__m256i)((__v4du){20, 30, 40, 50}),
                                                       (__m256i)((__v4du){30, 40, 50, 60})),
                          700, 1400, 2300, 3400));

TEST_CONSTEXPR(match_v4di(_mm256_maskz_madd52lo_epu64(0x9,
                                                       (__m256i)((__v4du){100, 200, 300, 400}),
                                                       (__m256i)((__v4du){10, 20, 30, 40}),
                                                       (__m256i)((__v4du){5, 10, 15, 20})),
                          150, 0, 0, 1200));

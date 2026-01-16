// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_madd52hi_epu64(__m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: test_mm512_madd52hi_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52h.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_madd52hi_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v8di(_mm512_madd52hi_epu64(
                              (__m512i)(__v8du){100, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){10, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){5, 0, 0, 0, 0, 0, 0, 0}),
                          100, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52hi_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull, 0, 0, 0,
                                                0, 0, 0, 0},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull, 0, 0, 0,
                                                0, 0, 0, 0}),
                          0xFFFFFFFFFFFFEull, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52hi_epu64(
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull}),
                          4503599627370495ull, 4503599627370496ull,
                          4503599627370497ull, 4503599627370498ull,
                          4503599627370499ull, 4503599627370500ull,
                          4503599627370501ull, 4503599627370502ull));

__m512i test_mm512_mask_madd52hi_epu64(__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_madd52hi_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52h.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_madd52hi_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v8di(_mm512_mask_madd52hi_epu64(
                              (__m512i)(__v8du){111, 222, 333, 444, 555, 666,
                                                777, 888},
                              0x00,
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80}),
                          111, 222, 333, 444, 555, 666, 777, 888));

TEST_CONSTEXPR(match_v8di(_mm512_mask_madd52hi_epu64(
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              0xFF,
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80}),
                          10, 20, 30, 40, 50, 60, 70, 80));

__m512i test_mm512_maskz_madd52hi_epu64(__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: test_mm512_maskz_madd52hi_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52h.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_madd52hi_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v8di(_mm512_maskz_madd52hi_epu64(
                              0x00,
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800}),
                          0, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_maskz_madd52hi_epu64(
                              0xFF,
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800}),
                          1, 2, 3, 4, 5, 6, 7, 8));

__m512i test_mm512_madd52lo_epu64(__m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: test_mm512_madd52lo_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52l.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_madd52lo_epu64(__X, __Y, __Z);
}

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){10, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){5, 0, 0, 0, 0, 0, 0, 0}),
                          50, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){100, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){20, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){30, 0, 0, 0, 0, 0, 0, 0}),
                          700, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull, 0, 0, 0,
                                                0, 0, 0, 0},
                              (__m512i)(__v8du){1, 0, 0, 0, 0, 0, 0, 0}),
                          0xFFFFFFFFFFFFFull, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){0x1F000000000000ull, 0, 0, 0,
                                                0, 0, 0, 0},
                              (__m512i)(__v8du){2, 0, 0, 0, 0, 0, 0, 0}),
                          0xE000000000000ull, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              (__m512i)(__v8du){2, 3, 4, 5, 6, 7, 8, 9}),
                          21, 62, 123, 204, 305, 426, 567, 728));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull, 0, 0, 0,
                                                0, 0, 0, 0},
                              (__m512i)(__v8du){10, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){5, 0, 0, 0, 0, 0, 0, 0}),
                          4503599627370545ull, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800},
                              (__m512i)(__v8du){2, 3, 4, 5, 6, 7, 8, 9}),
                          210, 620, 1230, 2040, 3050, 4260, 5670, 7280));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){0x1F000000000000ull,
                                                0x1F000000000000ull, 0, 0, 0,
                                                0, 0, 0},
                              (__m512i)(__v8du){2, 3, 0, 0, 0, 0, 0, 0}),
                          0xE000000000000ull, 0xD000000000000ull, 0, 0, 0, 0,
                          0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_madd52lo_epu64(
                              (__m512i)(__v8du){0, 0, 0, 0, 0, 0, 0, 0},
                              (__m512i)(__v8du){0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull,
                                                0xFFFFFFFFFFFFFull},
                              (__m512i)(__v8du){1, 1, 1, 1, 1, 1, 1, 1}),
                          0xFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFull,
                          0xFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFull,
                          0xFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFull,
                          0xFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFull));

__m512i test_mm512_mask_madd52lo_epu64(__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_madd52lo_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52l.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_madd52lo_epu64(__W, __M, __X, __Y);
}

TEST_CONSTEXPR(match_v8di(_mm512_mask_madd52lo_epu64(
                              (__m512i)(__v8du){111, 222, 333, 444, 555, 666,
                                                777, 888},
                              0x00,
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80}),
                          111, 222, 333, 444, 555, 666, 777, 888));

TEST_CONSTEXPR(match_v8di(_mm512_mask_madd52lo_epu64(
                              (__m512i)(__v8du){1000, 2000, 3000, 4000, 5000,
                                                6000, 7000, 8000},
                              0xFF,
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800},
                              (__m512i)(__v8du){20, 30, 40, 50, 60, 70, 80,
                                                90}),
                          3000, 8000, 15000, 24000, 35000, 48000, 63000,
                          80000));

__m512i test_mm512_maskz_madd52lo_epu64(__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: test_mm512_maskz_madd52lo_epu64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.vpmadd52l.uq.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_madd52lo_epu64(__M, __X, __Y, __Z);
}

TEST_CONSTEXPR(match_v8di(_mm512_maskz_madd52lo_epu64(
                              0x00,
                              (__m512i)(__v8du){1, 2, 3, 4, 5, 6, 7, 8},
                              (__m512i)(__v8du){10, 20, 30, 40, 50, 60, 70,
                                                80},
                              (__m512i)(__v8du){2, 3, 4, 5, 6, 7, 8, 9}),
                          0, 0, 0, 0, 0, 0, 0, 0));

TEST_CONSTEXPR(match_v8di(_mm512_maskz_madd52lo_epu64(
                              0xFF,
                              (__m512i)(__v8du){100, 200, 300, 400, 500, 600,
                                                700, 800},
                              (__m512i)(__v8du){20, 30, 40, 50, 60, 70, 80,
                                                90},
                              (__m512i)(__v8du){30, 40, 50, 60, 70, 80, 90,
                                                100}),
                          700, 1400, 2300, 3400, 4700, 6200, 7900, 9800));

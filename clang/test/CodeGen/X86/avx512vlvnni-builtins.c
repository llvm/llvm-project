// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m256i test_mm256_mask_dpbusd_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_dpbusd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbusd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_mask_dpbusd_epi32(
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__mmask8)0x55,
    (__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  104, 200, 304, 400, 504, 600, 704, 800));

__m256i test_mm256_maskz_dpbusd_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpbusd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbusd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpbusd_epi32(
    (__mmask8)0x0F,
    (__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0},
    (__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  4, 4, 4, 4, 0, 0, 0, 0));

__m256i test_mm256_dpbusd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpbusd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_dpbusd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusd_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  4, 4, 4, 4, 4, 4, 4, 4));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusd_epi32(
    ((__m256i)(__v8si){10, 10, 10, 10, 10, 10, 10, 10}),
    ((__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  14, 14, 14, 14, 14, 14, 14, 14));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusd_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v32qu){255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255}),
    ((__m256i)(__v32qi){-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1})),
  -1020, -1020, -1020, -1020, -1020, -1020, -1020, -1020));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusd_epi32(
    ((__m256i)(__v8si){2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m256i)(__v32qu){1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0}),
    ((__m256i)(__v32qi){1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m256i test_mm256_mask_dpbusds_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_dpbusds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbusds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_mask_dpbusds_epi32(
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__mmask8)0xAA,
    (__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  100, 204, 300, 404, 500, 604, 700, 804));

__m256i test_mm256_maskz_dpbusds_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpbusds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbusds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpbusds_epi32(
    (__mmask8)0xFF,
    (__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0},
    (__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  4, 4, 4, 4, 4, 4, 4, 4));

__m256i test_mm256_dpbusds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpbusds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_dpbusds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusds_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  4, 4, 4, 4, 4, 4, 4, 4));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusds_epi32(
    ((__m256i)(__v8si){2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m256i)(__v32qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v32qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusds_epi32(
    ((__m256i)(__v8si){-2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1}),
    ((__m256i)(__v32qu){255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255}),
    ((__m256i)(__v32qi){-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m256i test_mm256_mask_dpwssd_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_dpwssd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwssd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_mask_dpwssd_epi32(
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__mmask8)0xF0,
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  100, 200, 300, 400, 502, 602, 702, 802));

__m256i test_mm256_maskz_dpwssd_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpwssd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwssd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpwssd_epi32(
    (__mmask8)0x0F,
    (__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  2, 2, 2, 2, 0, 0, 0, 0));

__m256i test_mm256_dpwssd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpwssd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_dpwssd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  2, 2, 2, 2, 2, 2, 2, 2));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){10, 10, 10, 10, 10, 10, 10, 10}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  12, 12, 12, 12, 12, 12, 12, 12));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v16hi){-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  -2, -2, -2, -2, -2, -2, -2, -2));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v16hi){32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767}),
    ((__m256i)(__v16hi){32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767})),
  2147352578, 2147352578, 2147352578, 2147352578, 2147352578, 2147352578, 2147352578, 2147352578));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m256i)(__v16hi){1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0}),
    ((__m256i)(__v16hi){1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m256i test_mm256_mask_dpwssds_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_dpwssds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwssds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_mask_dpwssds_epi32(
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__mmask8)0xAA,
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  100, 202, 300, 402, 500, 602, 700, 802));

__m256i test_mm256_maskz_dpwssds_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpwssds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwssds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpwssds_epi32(
    (__mmask8)0xFF,
    (__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  2, 2, 2, 2, 2, 2, 2, 2));

__m256i test_mm256_dpwssds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpwssds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_dpwssds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssds_epi32(
    ((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m256i)(__v16hi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  2, 2, 2, 2, 2, 2, 2, 2));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssds_epi32(
    ((__m256i)(__v8si){2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m256i)(__v16hi){32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767}),
    ((__m256i)(__v16hi){32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767})),
  2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647));
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssds_epi32(
    ((__m256i)(__v8si){-2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1}),
    ((__m256i)(__v16hi){-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768}),
    ((__m256i)(__v16hi){32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767,32767})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m128i test_mm_mask_dpbusd_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbusd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpbusd_epi32(
    (__m128i)(__v4si){100, 200, 300, 400},
    (__mmask8)0x05,
    (__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  104, 200, 304, 400));

__m128i test_mm_maskz_dpbusd_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbusd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpbusd_epi32(
    (__mmask8)0x03,
    (__m128i)(__v4si){0, 0, 0, 0},
    (__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  4, 4, 0, 0));

__m128i test_mm_dpbusd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_dpbusd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  4, 4, 4, 4));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){10, 10, 10, 10}),
    ((__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  14, 14, 14, 14));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v16qu){255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255}),
    ((__m128i)(__v16qi){-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1})),
  -1020, -1020, -1020, -1020));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m128i)(__v16qu){1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0}),
    ((__m128i)(__v16qi){1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m128i test_mm_mask_dpbusds_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbusds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpbusds_epi32(
    (__m128i)(__v4si){100, 200, 300, 400},
    (__mmask8)0x0A,
    (__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  100, 204, 300, 404));

__m128i test_mm_maskz_dpbusds_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbusds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpbusds_epi32(
    (__mmask8)0x0F,
    (__m128i)(__v4si){0, 0, 0, 0},
    (__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    (__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
  4, 4, 4, 4));

__m128i test_mm_dpbusds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_dpbusds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusds_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  4, 4, 4, 4));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusds_epi32(
    ((__m128i)(__v4si){2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m128i)(__v16qu){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}),
    ((__m128i)(__v16qi){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1})),
  2147483647, 2147483647, 2147483647, 2147483647));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusds_epi32(
    ((__m128i)(__v4si){-2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1}),
    ((__m128i)(__v16qu){255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255}),
    ((__m128i)(__v16qi){-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m128i test_mm_mask_dpwssd_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwssd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpwssd_epi32(
    (__m128i)(__v4si){100, 200, 300, 400},
    (__mmask8)0x05,
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
  102, 200, 302, 400));

__m128i test_mm_maskz_dpwssd_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwssd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpwssd_epi32(
    (__mmask8)0x03,
    (__m128i)(__v4si){0, 0, 0, 0},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
  2, 2, 0, 0));

__m128i test_mm_dpwssd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_dpwssd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1})),
  2, 2, 2, 2));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){10, 10, 10, 10}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1})),
  12, 12, 12, 12));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v8hi){-1,-1,-1,-1,-1,-1,-1,-1}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1})),
  -2, -2, -2, -2));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v8hi){32767,32767,32767,32767,32767,32767,32767,32767}),
    ((__m128i)(__v8hi){32767,32767,32767,32767,32767,32767,32767,32767})),
  2147352578, 2147352578, 2147352578, 2147352578));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m128i)(__v8hi){1,0,1,0,1,0,1,0}),
    ((__m128i)(__v8hi){1,0,1,0,1,0,1,0})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));

__m128i test_mm_mask_dpwssds_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwssds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpwssds_epi32(
    (__m128i)(__v4si){100, 200, 300, 400},
    (__mmask8)0x0A,
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
  100, 202, 300, 402));

__m128i test_mm_maskz_dpwssds_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwssds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpwssds_epi32(
    (__mmask8)0x0F,
    (__m128i)(__v4si){0, 0, 0, 0},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1},
    (__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
  2, 2, 2, 2));

__m128i test_mm_dpwssds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_dpwssds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssds_epi32(
    ((__m128i)(__v4si){0, 0, 0, 0}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1}),
    ((__m128i)(__v8hi){1,1,1,1,1,1,1,1})),
  2, 2, 2, 2));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssds_epi32(
    ((__m128i)(__v4si){2147483647, 2147483647, 2147483647, 2147483647}),
    ((__m128i)(__v8hi){32767,32767,32767,32767,32767,32767,32767,32767}),
    ((__m128i)(__v8hi){32767,32767,32767,32767,32767,32767,32767,32767})),
  2147483647, 2147483647, 2147483647, 2147483647));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssds_epi32(
    ((__m128i)(__v4si){-2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1}),
    ((__m128i)(__v8hi){-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768}),
    ((__m128i)(__v8hi){32767,32767,32767,32767,32767,32767,32767,32767})),
  -2147483647-1, -2147483647-1, -2147483647-1, -2147483647-1));


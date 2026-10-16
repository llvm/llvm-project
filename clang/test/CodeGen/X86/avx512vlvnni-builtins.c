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
    (__m256i)(__v32qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32},
    (__m256i)(__v32qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16, -17,18,-19,20, -21,22,-23,24, -25,26,-27,28, -29,30,-31,32}),
  110, 200, 342, 400, 574, 600, 806, 800));

__m256i test_mm256_maskz_dpbusd_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpbusd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbusd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpbusd_epi32(
    (__mmask8)0x0F,
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__m256i)(__v32qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32},
    (__m256i)(__v32qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16, -17,18,-19,20, -21,22,-23,24, -25,26,-27,28, -29,30,-31,32}),
  110, 226, 342, 458, 0, 0, 0, 0));

__m256i test_mm256_dpbusd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpbusd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_dpbusd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusd_epi32(
    ((__m256i)(__v8si){2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m256i)(__v32qu){255,1,8,15, 255,22,6,13, 255,20,4,11, 255,18,2,9, 0,16,23,7, 0,14,21,5, 0,12,19,3, 0,10,17,1}),
    ((__m256i)(__v32qi){127,-6,-1,4, 127,9,-5,0, -128,5,-9,-4, -128,1,6,-8, 127,-3,2,7, 127,-7,-2,3, -128,8,-6,-1, -128,4,9,-5})),
  -2147451218, -2147451095, 2147451027, 2147450966, -2147483602, 2147483523, 2147483626, -2147483460));

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
    (__m256i)(__v32qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32},
    (__m256i)(__v32qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16, -17,18,-19,20, -21,22,-23,24, -25,26,-27,28, -29,30,-31,32}),
  100, 226, 300, 458, 500, 690, 700, 922));

__m256i test_mm256_maskz_dpbusds_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpbusds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbusds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpbusds_epi32(
    (__mmask8)0xFF,
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__m256i)(__v32qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32},
    (__m256i)(__v32qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16, -17,18,-19,20, -21,22,-23,24, -25,26,-27,28, -29,30,-31,32}),
  110, 226, 342, 458, 574, 690, 806, 922));

__m256i test_mm256_dpbusds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpbusds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_dpbusds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpbusds_epi32(
    ((__m256i)(__v8si){2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m256i)(__v32qu){255,1,8,15, 255,22,6,13, 255,20,4,11, 255,18,2,9, 0,16,23,7, 0,14,21,5, 0,12,19,3, 0,10,17,1}),
    ((__m256i)(__v32qi){127,-6,-1,4, 127,9,-5,0, -128,5,-9,-4, -128,1,6,-8, 127,-3,2,7, 127,-7,-2,3, -128,8,-6,-1, -128,4,9,-5})),
  2147483647, -2147451095, 2147451027, -2147483647-1, 2147483647, -2147483647-1, 2147483626, -2147483460));

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
    (__m256i)(__v16hi){1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16},
    (__m256i)(__v16hi){-1,2, -3,4, -5,6, -7,8, -9,10, -11,12, -13,14, -15,16}),
  100, 200, 300, 400, 519, 623, 727, 831));

__m256i test_mm256_maskz_dpwssd_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpwssd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwssd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpwssd_epi32(
    (__mmask8)0x0F,
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__m256i)(__v16hi){1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16},
    (__m256i)(__v16hi){-1,2, -3,4, -5,6, -7,8, -9,10, -11,12, -13,14, -15,16}),
  103, 207, 311, 415, 0, 0, 0, 0));

__m256i test_mm256_dpwssd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpwssd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_dpwssd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssd_epi32(
    ((__m256i)(__v8si){2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m256i)(__v16hi){32767,1, 32767,8, 32767,15, 32767,22, -32768,6, -32768,13, -32768,20, -32768,4}),
    ((__m256i)(__v16hi){32767,-6, 32767,-1, -32768,4, -32768,9, 32767,-5, 32767,0, -32768,5, -32768,-9})),
  -1073807366, -1073807367, 1073774651, 1073774790, 1073774561, 1073774592, -1073741725, -1073741860));

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
    (__m256i)(__v16hi){1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16},
    (__m256i)(__v16hi){-1,2, -3,4, -5,6, -7,8, -9,10, -11,12, -13,14, -15,16}),
  100, 207, 300, 415, 500, 623, 700, 831));

__m256i test_mm256_maskz_dpwssds_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dpwssds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwssds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_maskz_dpwssds_epi32(
    (__mmask8)0xFF,
    (__m256i)(__v8si){100, 200, 300, 400, 500, 600, 700, 800},
    (__m256i)(__v16hi){1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16},
    (__m256i)(__v16hi){-1,2, -3,4, -5,6, -7,8, -9,10, -11,12, -13,14, -15,16}),
  103, 207, 311, 415, 519, 623, 727, 831));

__m256i test_mm256_dpwssds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dpwssds_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_dpwssds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(
  _mm256_dpwssds_epi32(
    ((__m256i)(__v8si){2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m256i)(__v16hi){32767,1, 32767,8, 32767,15, 32767,22, -32768,6, -32768,13, -32768,20, -32768,4}),
    ((__m256i)(__v16hi){32767,-6, 32767,-1, -32768,4, -32768,9, 32767,-5, 32767,0, -32768,5, -32768,-9})),
  2147483647, -1073807367, 1073774651, -2147483647-1, 1073774561, -2147483647-1, 2147483647, -1073741860));

__m128i test_mm_mask_dpbusd_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbusd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpbusd_epi32(
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__mmask8)0x05,
    (__m128i)(__v16qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
    (__m128i)(__v16qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16}),
  1010, 2000, 3042, 4000));

__m128i test_mm_maskz_dpbusd_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbusd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpbusd_epi32(
    (__mmask8)0x03,
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__m128i)(__v16qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
    (__m128i)(__v16qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16}),
  1010, 2026, 0, 0));

__m128i test_mm_dpbusd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpbusd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_dpbusd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m128i)(__v16qu){255,1,8,15, 255,22,6,13, 0,20,4,11, 0,18,2,9}),
    ((__m128i)(__v16qi){127,-6,-1,4, -128,9,-5,0, 127,5,-9,-4, -128,1,6,-8})),
  -2147451218, 2147451176, -2147483629, 2147483606));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusd_epi32(
    ((__m128i)(__v4si){-2147483647-1, 2147483647, -2147483647-1, 2147483647}),
    ((__m128i)(__v16qu){255,1,8,15, 255,22,6,13, 0,20,4,11, 0,18,2,9}),
    ((__m128i)(__v16qi){127,-6,-1,4, -128,9,-5,0, 127,5,-9,-4, -128,1,6,-8})),
  -2147451217, 2147451175, -2147483628, 2147483605));

__m128i test_mm_mask_dpbusds_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbusds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpbusds_epi32(
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__mmask8)0x0A,
    (__m128i)(__v16qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
    (__m128i)(__v16qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16}),
  1000, 2026, 3000, 4058));

__m128i test_mm_maskz_dpbusds_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbusds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpbusds_epi32(
    (__mmask8)0x0F,
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__m128i)(__v16qu){1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
    (__m128i)(__v16qi){-1,2,-3,4, -5,6,-7,8, -9,10,-11,12, -13,14,-15,16}),
  1010, 2026, 3042, 4058));

__m128i test_mm_dpbusds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpbusds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_dpbusds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusds_epi32(
    ((__m128i)(__v4si){2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m128i)(__v16qu){255,1,8,15, 255,22,6,13, 0,20,4,11, 0,18,2,9}),
    ((__m128i)(__v16qi){127,-6,-1,4, -128,9,-5,0, 127,5,-9,-4, -128,1,6,-8})),
  2147483647, -2147483647-1, 2147483647, -2147483647-1));
TEST_CONSTEXPR(match_v4si(
  _mm_dpbusds_epi32(
    ((__m128i)(__v4si){-2147483647-1, 2147483647, -2147483647-1, 2147483647}),
    ((__m128i)(__v16qu){255,1,8,15, 255,22,6,13, 0,20,4,11, 0,18,2,9}),
    ((__m128i)(__v16qi){127,-6,-1,4, -128,9,-5,0, 127,5,-9,-4, -128,1,6,-8})),
  -2147451217, 2147451175, -2147483628, 2147483605));

__m128i test_mm_mask_dpwssd_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwssd_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpwssd_epi32(
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__mmask8)0x05,
    (__m128i)(__v8hi){1,2, 3,4, 5,6, 7,8},
    (__m128i)(__v8hi){-1,2, -3,4, -5,6, -7,8}),
  1003, 2000, 3011, 4000));

__m128i test_mm_maskz_dpwssd_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwssd_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpwssd_epi32(
    (__mmask8)0x03,
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__m128i)(__v8hi){1,2, 3,4, 5,6, 7,8},
    (__m128i)(__v8hi){-1,2, -3,4, -5,6, -7,8}),
  1003, 2007, 0, 0));

__m128i test_mm_dpwssd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpwssd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_dpwssd_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m128i)(__v8hi){32767,1, 32767,8, -32768,15, -32768,22}),
    ((__m128i)(__v8hi){32767,-6, -32768,-1, 32767,4, -32768,9})),
  -1073807366, 1073774584, 1073774651, -1073741626));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssd_epi32(
    ((__m128i)(__v4si){-2147483647-1, 2147483647, -2147483647-1, 2147483647}),
    ((__m128i)(__v8hi){32767,1, 32767,8, -32768,15, -32768,22}),
    ((__m128i)(__v8hi){32767,-6, -32768,-1, 32767,4, -32768,9})),
  -1073807365, 1073774583, 1073774652, -1073741627));

__m128i test_mm_mask_dpwssds_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwssds_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_mask_dpwssds_epi32(
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__mmask8)0x0A,
    (__m128i)(__v8hi){1,2, 3,4, 5,6, 7,8},
    (__m128i)(__v8hi){-1,2, -3,4, -5,6, -7,8}),
  1000, 2007, 3000, 4015));

__m128i test_mm_maskz_dpwssds_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwssds_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_maskz_dpwssds_epi32(
    (__mmask8)0x0F,
    (__m128i)(__v4si){1000, 2000, 3000, 4000},
    (__m128i)(__v8hi){1,2, 3,4, 5,6, 7,8},
    (__m128i)(__v8hi){-1,2, -3,4, -5,6, -7,8}),
  1003, 2007, 3011, 4015));

__m128i test_mm_dpwssds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dpwssds_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_dpwssds_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssds_epi32(
    ((__m128i)(__v4si){2147483647, -2147483647-1, 2147483647, -2147483647-1}),
    ((__m128i)(__v8hi){32767,1, 32767,8, -32768,15, -32768,22}),
    ((__m128i)(__v8hi){32767,-6, -32768,-1, 32767,4, -32768,9})),
  2147483647, -2147483647-1, 1073774651, -1073741626));
TEST_CONSTEXPR(match_v4si(
  _mm_dpwssds_epi32(
    ((__m128i)(__v4si){-2147483647-1, 2147483647, -2147483647-1, 2147483647}),
    ((__m128i)(__v8hi){32767,1, 32767,8, -32768,15, -32768,22}),
    ((__m128i)(__v8hi){32767,-6, -32768,-1, 32767,4, -32768,9})),
  -1073807365, 1073774583, -2147483647-1, 2147483647));

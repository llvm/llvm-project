// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

// NOTE: This should match the tests in llvm/test/CodeGen/X86/ssse3-intrinsics-fast-isel.ll

__m128i test_mm_abs_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi8
  // CHECK: [[ABS:%.*]] = call <16 x i8> @llvm.abs.v16i8(<16 x i8> %{{.*}}, i1 false)
  return _mm_abs_epi8(a);
}
TEST_CONSTEXPR(match_v16qi(_mm_abs_epi8((__m128i)(__v16qs){+100, +50, -100, +20, +80, -50, +120, -20, -100, -50, +100, -20, -80, +50, -120, +20}), 100, 50, 100, 20, 80, 50, 120, 20, 100, 50, 100, 20, 80, 50, 120, 20));

__m128i test_mm_abs_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi16
  // CHECK: [[ABS:%.*]] = call <8 x i16> @llvm.abs.v8i16(<8 x i16> %{{.*}}, i1 false)
  return _mm_abs_epi16(a);
}
TEST_CONSTEXPR(match_v8hi(_mm_abs_epi16((__m128i)(__v8hi){+32000, -32000, +6, -60, +80, -50, +120, -20}), 32000, 32000, 6, 60, 80, 50, 120, 20));

__m128i test_mm_abs_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_abs_epi32
  // CHECK: [[ABS:%.*]] = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %{{.*}}, i1 false)
  return _mm_abs_epi32(a);
}
TEST_CONSTEXPR(match_v4si(_mm_abs_epi32((__m128i)(__v4si){-5, -1, 0, 1}), 5, 1, 0, 1));

__m128i test_mm_alignr_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  return _mm_alignr_epi8(a, b, 2);
}

__m128i test2_mm_alignr_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test2_mm_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  return _mm_alignr_epi8(a, b, 17);
}

__m128i test_mm_hadd_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadd_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.w.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_hadd_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_hadd_epi16((__m128i)(__v8hi){1,2,3,4,5,6,7,8}, (__m128i)(__v8hi){17,18,19,20,21,22,23,24}), 3,7,11,15,35,39,43,47));

__m128i test_mm_hadd_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadd_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phadd.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_hadd_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_hadd_epi32((__m128i)(__v4si){1,2,3,4}, (__m128i)(__v4si){5,6,7,8}), 3,7,11,15));

__m128i test_mm_hadds_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hadds_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.sw.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_hadds_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_hadds_epi16((__m128i)(__v8hi){30000,30000,-1,2,-3,3,1,4}, (__m128i)(__v8hi){2,6,1,9,-4,16,7,8}), 32767, 1,0,5,8,10,12,15));


__m128i test_mm_hsub_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsub_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.w.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_hsub_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_hsub_epi16((__m128i)(__v8hi){20,15,16,12,9,6,4,2}, (__m128i)(__v8hi){3,2,1,1,4,5,0,2}), 5,4,3,2,1,0,-1,-2));

__m128i test_mm_hsub_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsub_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phsub.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_hsub_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_hsub_epi32((__m128i)(__v4si){4,3,1,1}, (__m128i)(__v4si){7,5,10,5}), 1,0,2,5));   

__m128i test_mm_hsubs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_hsubs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.sw.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_hsubs_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_hsubs_epi16((__m128i)(__v8hi){32767, -15,16,12,9,6,4,2},(__m128i)(__v8hi){3,2,1,1,4,5,0,2}), 32767,4,3,2,1,0,-1,-2));

__m128i test_mm_maddubs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_maddubs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_maddubs_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_maddubs_epi16((__m128i)(__v16qi){1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qs){2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -2, -2, -3, -3, -4, -4}), 5, 18, 39, 68, -3, -14, -33, -60));

__m128i test_mm_mulhrs_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_mulhrs_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mulhrs_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_mulhrs_epi16((__m128i)(__v8hi){+100, +200, -300, -400, +500, +600, -700, +800}, (__m128i)(__v8hi){+8000, -7000, +6000, -5000, +4000, -3000, +2000, -1000}), +24, -43, -55, +61, +61, -55, -43, -24));

__m128i test_mm_shuffle_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shuffle_epi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_shuffle_epi8(a, b);
}

TEST_CONSTEXPR(match_v16qi(_mm_shuffle_epi8((__m128i)(__v16qs){0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15}, (__m128i)(__v16qs){15,-14,13,-12,11,-10,9,-8,7,-6,5,-4,3,-2,1,0}), -15,0,-13,0,-11,0,-9,0,-7,0,-5,0,-3,0,-1,0));

__m128i test_mm_sign_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_sign_epi8(a, b);
}

__m128i test_mm_sign_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.psign.w.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_sign_epi16(a, b);
}

__m128i test_mm_sign_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sign_epi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.psign.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sign_epi32(a, b);
}

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <x86intrin.h>
#include "builtin_test_helpers.h"

// This test mimics clang/test/CodeGen/X86/xop-builtins.c, which eventually
// CIR shall be able to support fully.

__m128i test_mm_rot_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi8
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_rot_epi8(a, b);
}
TEST_CONSTEXPR(match_v16qi(_mm_rot_epi8((__m128i)(__v16qs){15, -14, -13, -12, 11, 10, 9, 8, 7, 6, 5, -4, 3, -2, 1, 0}, (__m128i)(__v16qs){0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15}), 15, -27, -4, -89, -80, 65, 36, 4, 7, 12, 65, -25, 48, -33, 4, 0));

__m128i test_mm_rot_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_rot_epi16(a, b);
}
TEST_CONSTEXPR(match_v8hi(_mm_rot_epi16((__m128i)(__v8hi){7, 6, 5, -4, 3, -2, 1, 0}, (__m128i)(__v8hi){0, 1, -2, 3, -4, 5, -6, 7}), 7, 12, 16385, -25, 12288, -33, 1024, 0));

__m128i test_mm_rot_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_rot_epi32(a, b);
}
TEST_CONSTEXPR(match_v4si(_mm_rot_epi32((__m128i)(__v4si){3, -2, 1, 0}, (__m128i)(__v4si){0, 1, -2, 3}), 3, -3, 1073741824, 0));

__m128i test_mm_rot_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_rot_epi64(a, b);
}
TEST_CONSTEXPR(match_v2di(_mm_rot_epi64((__m128i)(__v2di){99, -55}, (__m128i)(__v2di){1, -2}), 198, 9223372036854775794LL));

__m128i test_mm_roti_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi8
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> splat (i8 1))
  return _mm_roti_epi8(a, 1);
}
TEST_CONSTEXPR(match_v16qi(_mm_roti_epi8(((__m128i)(__v16qs){0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15}), 3), 0, 8, -9, 24, -25, 40, -41, 56, -57, 72, -73, 88, -89, 104, -105, 120));

__m128i test_mm_roti_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 50))
  return _mm_roti_epi16(a, 50);
}
TEST_CONSTEXPR(match_v8hi(_mm_roti_epi16(((__m128i)(__v8hi){2, -3, 4, -5, 6, -7, 8, -9}), 1), 4, -5, 8, -9, 12, -13, 16, -17));

__m128i test_mm_roti_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 226))
  return _mm_roti_epi32(a, -30);
}
TEST_CONSTEXPR(match_v4si(_mm_roti_epi32(((__m128i)(__v4si){1, -2, 3, -4}), 5), 32, -33, 96, -97));

__m128i test_mm_roti_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 100))
  return _mm_roti_epi64(a, 100);
}
TEST_CONSTEXPR(match_v2di(_mm_roti_epi64(((__m128i)(__v2di){99, -55}), 19), 51904512, -28311553));



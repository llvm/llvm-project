// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/avx-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m256 test_mm256_undefined_ps(void) {
  // CIR-LABEL: _mm256_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<8 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_undefined_ps
  // LLVM: store <8 x float> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <8 x float>, ptr %[[A]], align 32
  // LLVM: ret <8 x float> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_ps
  // OGCG: ret <8 x float> zeroinitializer
  return _mm256_undefined_ps();
}

__m256d test_mm256_undefined_pd(void) {
  // CIR-LABEL: _mm256_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_undefined_pd
  // LLVM: store <4 x double> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <4 x double>, ptr %[[A]], align 32
  // LLVM: ret <4 x double> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_pd
  // OGCG: ret <4 x double> zeroinitializer
  return _mm256_undefined_pd();
}

__m256i test_mm256_undefined_si256(void) {
  // CIR-LABEL: _mm256_undefined_si256
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<4 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_mm256_undefined_si256
  // LLVM: store <4 x i64> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <4 x i64>, ptr %[[A]], align 32
  // LLVM: ret <4 x i64> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_si256
  // OGCG: ret <4 x i64> zeroinitializer
  return _mm256_undefined_si256();
}

__m256d test_mm256_blend_pd(__m256d A, __m256d B) {
  // CIR-LABEL: test_mm256_blend_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>) [#cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_blend_pd
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>

  // OGCG-LABEL: test_mm256_blend_pd
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm256_blend_pd(A, B, 0x05);
}

__m256 test_mm256_blend_ps(__m256 A, __m256 B) {
  // CIR-LABEL: test_mm256_blend_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_blend_ps
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>

  // OGCG-LABEL: test_mm256_blend_ps
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_ps(A, B, 0x35);
}

__m256d test_mm256_insertf128_pd(__m256d A, __m128d B) {
  // CIR-LABEL: test_mm256_insertf128_pd
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_insertf128_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>

  // OGCG-LABEL: test_mm256_insertf128_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_insertf128_pd(A, B, 0);
}

__m256 test_mm256_insertf128_ps(__m256 A, __m128 B) {
  // CIR-LABEL: test_mm256_insertf128_ps
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.float>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_insertf128_ps
  // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>

  // OGCG-LABEL: test_mm256_insertf128_ps
  // OGCG: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf128_ps(A, B, 1);
}

__m256i test_mm256_insertf128_si256(__m256i A, __m128i B) {
  // CIR-LABEL: test_mm256_insertf128_si256
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !s32i>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s32i>) [#cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]

  // LLVM-LABEL: test_mm256_insertf128_si256
  // LLVM: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm256_insertf128_si256
  // OGCG: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  return _mm256_insertf128_si256(A, B, 0);
}

__m256d test_mm256_shuffle_pd(__m256d A, __m256d B) {
  // CIR-LABEL: test_mm256_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<2> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_shuffle_pd
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>

  // OGCG-LABEL: test_mm256_shuffle_pd
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_shuffle_pd(A, B, 0);
}

__m256 test_mm256_shuffle_ps(__m256 A, __m256 B) {
  // CIR-LABEL: test_mm256_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<8> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<12> : !s32i] : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_shuffle_ps
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 8, i32 8, i32 4, i32 4, i32 12, i32 12>

  // OGCG-LABEL: test_mm256_shuffle_ps
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 8, i32 8, i32 4, i32 4, i32 12, i32 12>
  return _mm256_shuffle_ps(A, B, 0);
}

__m128 test_mm_permute_ps(__m128 A) {
    // CIR-LABEL: test_mm_permute_ps
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} :  !cir.vector<4 x !cir.float>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !cir.float>

	// LLVM-LABEL: test_mm_permute_ps
    // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 0, i32 1>

    // OGCG-LABEL: test_mm_permute_ps
	// OGCG: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
    return _mm_permute_ps(A, 0x4E);
}

__m256 test_mm256_permute_ps(__m256 A) {
    // CIR-LABEL: test_mm256_permute_ps
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} :  !cir.vector<8 x !cir.float>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<8 x !cir.float>

    // LLVM-LABEL: test_mm256_permute_ps
    // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>

    // OGCG-LABEL: test_mm256_permute_ps
    // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
    return _mm256_permute_ps(A, 0x4E);
}

__m128d test_mm_permute_pd(__m128d A) {
    // CIR-LABEL: test_mm_permute_pd
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} :  !cir.vector<2 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.double>

    // LLVM-LABEL: test_mm_permute_pd
    // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 0>

    // OGCG-LABEL: test_mm_permute_pd
    // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 0>
    return _mm_permute_pd(A, 0x1);
}

__m256d test_mm256_permute_pd(__m256d A) {
    // CIR-LABEL: test_mm256_permute_pd
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} :  !cir.vector<4 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i] : !cir.vector<4 x !cir.double>

    // LLVM-LABEL: test_mm256_permute_pd
    // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>

    // OGCG-LABEL: test_mm256_permute_pd
    // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
    return _mm256_permute_pd(A, 0x5);
}

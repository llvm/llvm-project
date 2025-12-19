// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m256d test_mm256_insertf64x2(__m256d __A, __m128d __B) {
  // CIR-LABEL: test_mm256_insertf64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: @test_mm256_insertf64x2
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>

  // OGCG-LABEL: @test_mm256_insertf64x2
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_insertf64x2(__A, __B, 1);
}

__m256i test_mm256_inserti64x2(__m256i __A, __m128i __B) {
  // CIR-LABEL: test_mm256_inserti64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test_mm256_inserti64x2
  // LLVM: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>

  // OGCG-LABEL: @test_mm256_inserti64x2
  // OGCG: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti64x2(__A, __B, 1);
}

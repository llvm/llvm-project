// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=OGCG

#include <immintrin.h>

__m512i test_mm512_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm512_movm_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 8> -> !cir.vector<!s64i x 8>
  // LLVM-LABEL: @test_mm512_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i64>
  return _mm512_movm_epi64(__A); 
}

__m512 test_mm512_insertf32x8(__m512 __A, __m256 __B) {
  // CIR-LABEL: test_mm512_insertf32x8
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 16>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i] : !cir.vector<!cir.float x 16>

  // LLVM-LABEL: @test_mm512_insertf32x8
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  return _mm512_insertf32x8(__A, __B, 1); 
}

__m512i test_mm512_inserti32x8(__m512i __A, __m256i __B) {
  // CIR-LABEL: test_mm512_inserti32x8
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 16>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i] : !cir.vector<!s32i x 16>

  // LLVM-LABEL: @test_mm512_inserti32x8
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  return _mm512_inserti32x8(__A, __B, 1); 
}

__m512d test_mm512_insertf64x2(__m512d __A, __m128d __B) {
  // CIR-LABEL: test_mm512_insertf64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i] : !cir.vector<!cir.double x 8>

  // LLVM-LABEL: @test_mm512_insertf64x2
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  return _mm512_insertf64x2(__A, __B, 3); 
}

__m512i test_mm512_inserti64x2(__m512i __A, __m128i __B) {
  // CIR-LABEL: test_mm512_inserti64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s64i x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!s64i x 8>

  // LLVM-LABEL: @test_mm512_inserti64x2
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>
  return _mm512_inserti64x2(__A, __B, 1); 
}

__mmask16 test_mm512_movepi32_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi32_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s32i x 16>, !cir.vector<!cir.int<u, 1> x 16>

  // LLVM-LABEL: @test_mm512_movepi32_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm512_movepi32_mask
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer
  return _mm512_movepi32_mask(__A); 
}

__mmask8 test_mm512_movepi64_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi64_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s64i x 8>, !cir.vector<!cir.int<u, 1> x 8>

  // LLVM-LABEL: @test_mm512_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm512_movepi64_mask
  // OGCG: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer
  return _mm512_movepi64_mask(__A); 
}

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// FIXME: CIR to LLVM lowering fails with "integer width of the output type is smaller or equal to the integer width of the input type" error
// RUN-DISABLED: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN-DISABLED: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>


__m128i test_mm_movm_epi32(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi32
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<u, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<4 x !cir.int<u, 1>> -> !cir.vector<4 x !s32i>

  // LLVM-LABEL: @test_mm_movm_epi32
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i32>

  // OGCG-LABEL: @test_mm_movm_epi32
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_movm_epi32(__A); 
}

__m256i test_mm256_movm_epi32(__mmask8 __A) {
  // CIR-LABEL: _mm256_movm_epi32
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !cir.vector<8 x !s32i>

  // LLVM-LABEL: @test_mm256_movm_epi32
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i32>

  // OGCG-LABEL: @test_mm256_movm_epi32
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i32>
  return _mm256_movm_epi32(__A); 
}

__m512i test_mm512_movm_epi32(__mmask16 __A) {
  // CIR-LABEL: _mm512_movm_epi32
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !cir.vector<16 x !s32i>

  // LLVM-LABEL: @test_mm512_movm_epi32
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i32>

  // OGCG-LABEL: @test_mm512_movm_epi32
  // OGCG: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i32>
  return _mm512_movm_epi32(__A); 
}

__m128i test_mm_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<u, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<2 x !cir.int<u, 1>> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: @test_mm_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // LLVM: %{{.*}} = sext <2 x i1> %{{.*}} to <2 x i64>

  // OGCG-LABEL: @test_mm_movm_epi64
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // OGCG: %{{.*}} = sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_movm_epi64(__A); 
}

__m256i test_mm256_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm256_movm_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<u, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<4 x !cir.int<u, 1>> -> !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test_mm256_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i64>

  // OGCG-LABEL: @test_mm256_movm_epi64
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i64>
  return _mm256_movm_epi64(__A); 
}

__mmask8 test_mm256_movepi32_mask(__m256i __A) {
  // CIR-LABEL: _mm256_movepi32_mask
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !s32i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: @test_mm256_movepi32_mask
  // LLVM: [[CMP:%.*]] = icmp slt <8 x i32> %{{.*}}, zeroinitializer
  // LLVM: bitcast <8 x i1> [[CMP]] to i8

  // OGCG-LABEL: @test_mm256_movepi32_mask
  // OGCG: [[CMP:%.*]] = icmp slt <8 x i32> %{{.*}}, zeroinitializer
  // OGCG: bitcast <8 x i1> [[CMP]] to i8
  return _mm256_movepi32_mask(__A); 
}

__mmask8 test_mm_movepi64_mask(__m128i __A) {
  // CIR-LABEL: _mm_movepi64_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: @test_mm_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <2 x i64> %{{.*}}, zeroinitializer
  // LLVM: [[SHUF:%.*]] = shufflevector <2 x i1> [[CMP]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // LLVM: bitcast <8 x i1> [[SHUF]] to i8

  // OGCG-LABEL: @test_mm_movepi64_mask
  // OGCG: [[CMP:%.*]] = icmp slt <2 x i64> %{{.*}}, zeroinitializer
  // OGCG: [[SHUF:%.*]] = shufflevector <2 x i1> [[CMP]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG: bitcast <8 x i1> [[SHUF]] to i8
  return _mm_movepi64_mask(__A); 
}

__mmask8 test_mm256_movepi64_mask(__m256i __A) {
  // CIR-LABEL: _mm256_movepi64_mask
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !s64i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: [[SHUF:%.*]] = cir.vec.shuffle([[CMP]], %{{.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[SHUF]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: @test_mm256_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <4 x i64> %{{.*}}, zeroinitializer
  // LLVM: [[SHUF:%.*]] = shufflevector <4 x i1> [[CMP]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: bitcast <8 x i1> [[SHUF]] to i8

  // OGCG-LABEL: @test_mm256_movepi64_mask
  // OGCG: [[CMP:%.*]] = icmp slt <4 x i64> %{{.*}}, zeroinitializer
  // OGCG: [[SHUF:%.*]] = shufflevector <4 x i1> [[CMP]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: bitcast <8 x i1> [[SHUF]] to i8
  return _mm256_movepi64_mask(__A); 
}


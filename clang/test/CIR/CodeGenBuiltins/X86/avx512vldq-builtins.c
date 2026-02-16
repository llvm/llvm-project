// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>


__m128i test_mm_movm_epi32(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi32
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<4 x !cir.int<s, 1>> -> !cir.vector<4 x !s32i>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !cir.vector<8 x !s32i>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !cir.vector<16 x !s32i>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<2 x !cir.int<s, 1>> -> !cir.vector<2 x !s64i>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<4 x !cir.int<s, 1>> -> !cir.vector<4 x !s64i>

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


__mmask8 test_mm_mask_fpclass_pd_mask(__mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_fpclass_pd_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.pd.128"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[SHUF:.*]] = cir.vec.shuffle(%[[B]], %[[B]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.binop(and, %[[A]], %[[SHUF]]) : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %[[D:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %[[E:.*]] = cir.vec.shuffle(%[[C]], %[[D]] : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[E]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_mask_fpclass_pd_mask
  // LLVM: %[[A:.*]] = call <2 x i1> @llvm.x86.avx512.fpclass.pd.128
  // LLVM: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[C:.*]] = shufflevector <8 x i1> %[[B]], <8 x i1> %[[B]], <2 x i32> <i32 0, i32 1>
  // LLVM: %[[D:.*]] = and <2 x i1> %[[A]], %[[C]]
  // LLVM: %[[E:.*]] = shufflevector <2 x i1> %[[D]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // LLVM: bitcast <8 x i1> %[[E]] to i8

  // OGCG-LABEL: test_mm_mask_fpclass_pd_mask
  // OGCG: %[[A:.*]] = call <2 x i1> @llvm.x86.avx512.fpclass.pd.128
  // OGCG: and <2 x i1>
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm_mask_fpclass_pd_mask(__U, __A, 2);
}

__mmask8 test_mm_fpclass_pd_mask(__m128d __A) {
  // CIR-LABEL: _mm_fpclass_pd_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.pd.128"
  // CIR: %[[B:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.vec.shuffle(%[[A]], %[[B]] : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_fpclass_pd_mask
  // LLVM: %[[A:.*]] = call <2 x i1> @llvm.x86.avx512.fpclass.pd.128
  // LLVM: %[[B:.*]] = shufflevector <2 x i1> %[[A]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // LLVM: bitcast <8 x i1> %[[B]] to i8

  // OGCG-LABEL: test_mm_fpclass_pd_mask
  // OGCG: %[[A:.*]] = call <2 x i1> @llvm.x86.avx512.fpclass.pd.128
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm_fpclass_pd_mask(__A, 2);
}

__mmask8 test_mm256_mask_fpclass_pd_mask(__mmask8 __U, __m256d __A) {
  // CIR-LABEL: _mm256_mask_fpclass_pd_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.pd.256"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[SHUF:.*]] = cir.vec.shuffle(%[[B]], %[[B]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.binop(and, %[[A]], %[[SHUF]]) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[D:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[E:.*]] = cir.vec.shuffle(%[[C]], %[[D]] : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[E]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm256_mask_fpclass_pd_mask
  // LLVM: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256
  // LLVM: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[C:.*]] = shufflevector <8 x i1> %[[B]], <8 x i1> %[[B]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[D:.*]] = and <4 x i1> %[[A]], %[[C]]
  // LLVM: %[[E:.*]] = shufflevector <4 x i1> %[[D]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: bitcast <8 x i1> %[[E]] to i8

  // OGCG-LABEL: test_mm256_mask_fpclass_pd_mask
  // OGCG: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256
  // OGCG: and <4 x i1>
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm256_mask_fpclass_pd_mask(__U, __A, 2);
}

__mmask8 test_mm256_fpclass_pd_mask(__m256d __A) {
  // CIR-LABEL: _mm256_fpclass_pd_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.pd.256"
  // CIR: %[[B:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.vec.shuffle(%[[A]], %[[B]] : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm256_fpclass_pd_mask
  // LLVM: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256
  // LLVM: %[[B:.*]] = shufflevector <4 x i1> %[[A]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: bitcast <8 x i1> %[[B]] to i8

  // OGCG-LABEL: test_mm256_fpclass_pd_mask
  // OGCG: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm256_fpclass_pd_mask(__A, 2);
}

__mmask8 test_mm_mask_fpclass_ps_mask(__mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_fpclass_ps_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.ps.128"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[SHUF:.*]] = cir.vec.shuffle(%[[B]], %[[B]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.binop(and, %[[A]], %[[SHUF]]) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[D:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[E:.*]] = cir.vec.shuffle(%[[C]], %[[D]] : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[E]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_mask_fpclass_ps_mask
  // LLVM: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.ps.128
  // LLVM: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[C:.*]] = shufflevector <8 x i1> %[[B]], <8 x i1> %[[B]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[D:.*]] = and <4 x i1> %[[A]], %[[C]]
  // LLVM: %[[E:.*]] = shufflevector <4 x i1> %[[D]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: bitcast <8 x i1> %[[E]] to i8

  // OGCG-LABEL: test_mm_mask_fpclass_ps_mask
  // OGCG: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.ps.128
  // OGCG: and <4 x i1>
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm_mask_fpclass_ps_mask(__U, __A, 2);
}

__mmask8 test_mm_fpclass_ps_mask(__m128 __A) {
  // CIR-LABEL: _mm_fpclass_ps_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.ps.128"
  // CIR: %[[B:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.vec.shuffle(%[[A]], %[[B]] : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_fpclass_ps_mask
  // LLVM: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.ps.128
  // LLVM: %[[B:.*]] = shufflevector <4 x i1> %[[A]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: bitcast <8 x i1> %[[B]] to i8

  // OGCG-LABEL: test_mm_fpclass_ps_mask
  // OGCG: %[[A:.*]] = call <4 x i1> @llvm.x86.avx512.fpclass.ps.128
  // OGCG: shufflevector
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  return _mm_fpclass_ps_mask(__A, 2);
}

__mmask8 test_mm256_mask_fpclass_ps_mask(__mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_fpclass_ps_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.ps.256"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.binop(and, %[[A]], %[[B]]) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm256_mask_fpclass_ps_mask
  // LLVM: %[[A:.*]] = call <8 x i1> @llvm.x86.avx512.fpclass.ps.256
  // LLVM: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[C:.*]] = and <8 x i1> %[[A]], %[[B]]
  // LLVM: bitcast <8 x i1> %[[C]] to i8

  // OGCG-LABEL: test_mm256_mask_fpclass_ps_mask
  // OGCG: %[[A:.*]] = call <8 x i1> @llvm.x86.avx512.fpclass.ps.256
  // OGCG: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // OGCG: %[[C:.*]] = and <8 x i1> %[[A]], %[[B]]
  // OGCG: bitcast <8 x i1> %[[C]] to i8
  return _mm256_mask_fpclass_ps_mask(__U, __A, 2);
}

__mmask8 test_mm256_fpclass_ps_mask(__m256 __A) {
  // CIR-LABEL: _mm256_fpclass_ps_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx512.fpclass.ps.256"
  // CIR: cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm256_fpclass_ps_mask
  // LLVM: %[[A:.*]] = call <8 x i1> @llvm.x86.avx512.fpclass.ps.256
  // LLVM: bitcast <8 x i1> %[[A]] to i8

  // OGCG-LABEL: test_mm256_fpclass_ps_mask
  // OGCG: %[[A:.*]] = call <8 x i1> @llvm.x86.avx512.fpclass.ps.256
  // OGCG: bitcast <8 x i1> %[[A]] to i8
  return _mm256_fpclass_ps_mask(__A, 2);
}


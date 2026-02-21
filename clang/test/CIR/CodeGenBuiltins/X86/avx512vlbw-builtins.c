// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1-512 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG


#include <immintrin.h>

__m128i test_mm_movm_epi8(__mmask16 __A) {
  // CIR-LABEL: _mm_movm_epi8
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !cir.vector<16 x !s8i>

  // LLVM-LABEL: @test_mm_movm_epi8
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i8>

  // OGCG-LABEL: @test_mm_movm_epi8
  // OGCG: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_movm_epi8(__A);
}

__m256i test_mm256_movm_epi8(__mmask32 __A) {
  // CIR-LABEL: _mm256_movm_epi8
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !cir.vector<32 x !s8i>

  // LLVM-LABEL: @test_mm256_movm_epi8
  // LLVM: %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i8>

  // OGCG-LABEL: @test_mm256_movm_epi8
  // OGCG: %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i8>
  return _mm256_movm_epi8(__A);
}

__m512i test_mm512_movm_epi8(__mmask64 __A) {
  // CIR-LABEL: _mm512_movm_epi8
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !cir.vector<64 x !s8i>

  // LLVM-LABEL: @test_mm512_movm_epi8
  // LLVM:  %{{.*}} = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM:  %{{.*}} = sext <64 x i1> %{{.*}} to <64 x i8>

  // OGCG-LABEL: @test_mm512_movm_epi8
  // OGCG: %{{.*}} = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %{{.*}} = sext <64 x i1> %{{.*}} to <64 x i8>
  return _mm512_movm_epi8(__A);
}

__m128i test_mm_movm_epi16(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi16
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !cir.vector<8 x !s16i>

  // LLVM-LABEL: @test_mm_movm_epi16
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i16>

  // OGCG-LABEL: @test_mm_movm_epi16
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_movm_epi16(__A);
}

__m256i test_mm256_movm_epi16(__mmask16 __A) {
  // CIR-LABEL: _mm256_movm_epi16
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !cir.vector<16 x !s16i>

  // LLVM-LABEL: @test_mm256_movm_epi16
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i16>

  // OGCG-LABEL: @test_mm256_movm_epi16
  // OGCG: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i16>
  return _mm256_movm_epi16(__A);
}

__mmask16 test_mm_movepi8_mask(__m128i __A) {
  // CIR-LABEL: _mm_movepi8_mask
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<16 x !s8i>
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: @test_mm_movepi8_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer
  // LLVM: bitcast <16 x i1> [[CMP]] to i16

  // OGCG-LABEL: @test_mm_movepi8_mask
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer
  // OGCG: bitcast <16 x i1> [[CMP]] to i16
  return _mm_movepi8_mask(__A);
}

__mmask16 test_mm256_movepi16_mask(__m256i __A) {
  // CIR-LABEL: _mm256_movepi16_mask
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s64i> -> !cir.vector<16 x !s16i>
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !s16i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: @test_mm256_movepi16_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  // LLVM: bitcast <16 x i1> [[CMP]] to i16

  // OGCG-LABEL: @test_mm256_movepi16_mask
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  // OGCG: bitcast <16 x i1> [[CMP]] to i16
  return _mm256_movepi16_mask(__A);
}
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM-UNSIGNED-CHAR --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1-512 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG


#include <immintrin.h>

__m128i test_mm_movm_epi8(__mmask16 __A) {
  // CIR-LABEL: _mm_movm_epi8
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<!cir.int<s, 1> x 16>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 16> -> !cir.vector<{{!s8i|!u8i}} x 16>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<!cir.int<s, 1> x 32>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 32> -> !cir.vector<{{!s8i|!u8i}} x 32>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<!cir.int<s, 1> x 64>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 64> -> !cir.vector<{{!s8i|!u8i}} x 64>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 8> -> !cir.vector<!s16i x 8>

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
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<!cir.int<s, 1> x 16>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 16> -> !cir.vector<!s16i x 16>

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
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<{{!s8i|!u8i}} x 16>, !cir.vector<!cir.int<u, 1> x 16>

  // LLVM-LABEL: @test_mm_movepi8_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer

  // In the unsigned case below, the canonicalizer proves the comparison is
  // always false (no i8 unsigned value can be < 0) and folds it away.
  // LLVM-UNSIGNED-CHAR: store i16 0, ptr %{{.*}}, align 2

  // OGCG-LABEL: @test_mm_movepi8_mask
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer
  return _mm_movepi8_mask(__A); 
}

__mmask16 test_mm256_movepi16_mask(__m256i __A) {
  // CIR-LABEL: _mm256_movepi16_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s16i x 16>, !cir.vector<!cir.int<u, 1> x 16>

  // LLVM-LABEL: @test_mm256_movepi16_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm256_movepi16_mask
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  return _mm256_movepi16_mask(__A); 
}
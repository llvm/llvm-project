// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char  -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux  -target-feature +avx512bw -fno-signed-char  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM-UNSIGNED-CHAR --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

void test_mm512_mask_storeu_epi16(void *__P, __mmask32 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s16i x 32>, !cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>) -> !void

  // LLVM-LABEL: @test_mm512_mask_storeu_epi16
  // LLVM: call void @llvm.masked.store.v32i16.p0(<32 x i16> %{{.*}}, ptr elementtype(<32 x i16>) align 1 %{{.*}}, <32 x i1> %{{.*}})
  return _mm512_mask_storeu_epi16(__P, __U, __A);
}

void test_mm512_mask_storeu_epi8(void *__P, __mmask64 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<{{!s8i|!u8i}} x 64>, !cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>) -> !void

  // LLVM-LABEL: @test_mm512_mask_storeu_epi8
  // LLVM: call void @llvm.masked.store.v64i8.p0(<64 x i8> %{{.*}}, ptr elementtype(<64 x i8>) align 1 %{{.*}}, <64 x i1> %{{.*}})
  return _mm512_mask_storeu_epi8(__P, __U, __A); 
}

__m512i test_mm512_movm_epi16(__mmask32 __A) {
  // CIR-LABEL: _mm512_movm_epi16
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<!cir.int<s, 1> x 32>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<!cir.int<s, 1> x 32> -> !cir.vector<!s16i x 32>
  // LLVM-LABEL: @test_mm512_movm_epi16
  // LLVM:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM:  %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i16>
  return _mm512_movm_epi16(__A); 
}

__m512i test_mm512_mask_loadu_epi8(__m512i __W, __mmask64 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_loadu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>, !cir.vector<{{!s8i|!u8i}} x 64>) -> !cir.vector<{{!s8i|!u8i}} x 64>

  // LLVM-LABEL: @test_mm512_mask_loadu_epi8
  // LLVM: @llvm.masked.load.v64i8.p0(ptr elementtype(<64 x i8>) align 1 %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_mask_loadu_epi8(__W, __U, __P); 
}

__m512i test_mm512_mask_loadu_epi16(__m512i __W, __mmask32 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_loadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s16i x 32>) -> !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_mask_loadu_epi16
  // LLVM: @llvm.masked.load.v32i16.p0(ptr elementtype(<32 x i16>) align 1 %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_mask_loadu_epi16(__W, __U, __P); 
}

__m512i test_mm512_maskz_loadu_epi16(__mmask32 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_loadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s16i x 32>) -> !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_maskz_loadu_epi16
  // LLVM: @llvm.masked.load.v32i16.p0(ptr elementtype(<32 x i16>) align 1 %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_maskz_loadu_epi16(__U, __P); 
}

__m512i test_mm512_maskz_loadu_epi8(__mmask64 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_loadu_epi8
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>, !cir.vector<{{!s8i|!u8i}} x 64>) -> !cir.vector<{{!s8i|!u8i}} x 64>

  // LLVM-LABEL: @test_mm512_maskz_loadu_epi8
  // LLVM: @llvm.masked.load.v64i8.p0(ptr elementtype(<64 x i8>) align 1 %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_maskz_loadu_epi8(__U, __P); 
}

__mmask64 test_mm512_movepi8_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi8_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<{{!s8i|!u8i}} x 64>, !cir.vector<!cir.int<u, 1> x 64>

  // LLVM-LABEL: @test_mm512_movepi8_mask
  // LLVM: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer

  // In the unsigned case below, the canonicalizer proves the comparison is
  // always false (no i8 unsigned value can be < 0) and folds it away.
  // LLVM-UNSIGNED-CHAR: store i64 0, ptr %{{.*}}, align 8
  
  // OGCG-LABEL: @test_mm512_movepi8_mask
  // OGCG: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer
  return _mm512_movepi8_mask(__A); 
}

__mmask32 test_mm512_movepi16_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi16_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s16i x 32>, !cir.vector<!cir.int<u, 1> x 32>

  // LLVM-LABEL: @test_mm512_movepi16_mask
  // LLVM: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm512_movepi16_mask
  // OGCG: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer
  return _mm512_movepi16_mask(__A); 
}

__m512i test_mm512_shufflelo_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s16i x 32>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<9> : !s32i, #cir.int<8> : !s32i, #cir.int<8> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<17> : !s32i, #cir.int<16> : !s32i, #cir.int<16> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<25> : !s32i, #cir.int<24> : !s32i, #cir.int<24> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_shufflelo_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>

  // OGCG-LABEL: @test_mm512_shufflelo_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  return _mm512_shufflelo_epi16(__A, 5); 
}

__m512i test_mm512_shufflehi_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s16i x 32>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<13> : !s32i, #cir.int<12> : !s32i, #cir.int<12> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<21> : !s32i, #cir.int<20> : !s32i, #cir.int<20> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<29> : !s32i, #cir.int<28> : !s32i, #cir.int<28> : !s32i] : !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_shufflehi_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>

  // OGCG-LABEL: @test_mm512_shufflehi_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  return _mm512_shufflehi_epi16(__A, 5); 
}

__m512i test_mm512_alignr_epi8(__m512i __A,__m512i __B){
  // CIR-LABEL: _mm512_alignr_epi8
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<{{!s8i|!u8i}} x 64>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<96> : !s32i, #cir.int<97> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<112> : !s32i, #cir.int<113> : !s32i] : !cir.vector<{{!s8i|!u8i}} x 64>
    
  // LLVM-LABEL: @test_mm512_alignr_epi8
  // LLVM: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>

  // OGCG-LABEL: @test_mm512_alignr_epi8
  // OGCG: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>
  return _mm512_alignr_epi8(__A, __B, 2); 
}

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char  -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux  -target-feature +avx512bw -fno-signed-char  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m512i test_mm512_shufflelo_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<32 x !s16i>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<9> : !s32i, #cir.int<8> : !s32i, #cir.int<8> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<17> : !s32i, #cir.int<16> : !s32i, #cir.int<16> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<25> : !s32i, #cir.int<24> : !s32i, #cir.int<24> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<32 x !s16i>

  // LLVM-LABEL: @test_mm512_shufflelo_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>

  // OGCG-LABEL: @test_mm512_shufflelo_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  return _mm512_shufflelo_epi16(__A, 5);
}

__m512i test_mm512_shufflehi_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<32 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<13> : !s32i, #cir.int<12> : !s32i, #cir.int<12> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<21> : !s32i, #cir.int<20> : !s32i, #cir.int<20> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<29> : !s32i, #cir.int<28> : !s32i, #cir.int<28> : !s32i] : !cir.vector<32 x !s16i>

  // LLVM-LABEL: @test_mm512_shufflehi_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>

  // OGCG-LABEL: @test_mm512_shufflehi_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  return _mm512_shufflehi_epi16(__A, 5);
}

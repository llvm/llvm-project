// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

// This test exercises the kunpck (mask unpack) builtins for AVX-512.

#include <immintrin.h>

__mmask16 test_mm512_kunpackb(__mmask16 __A, __mmask16 __B) {
  // CIR-LABEL: test_mm512_kunpackb
  // LLVM-LABEL: test_mm512_kunpackb
  // OGCG-LABEL: test_mm512_kunpackb
  return _mm512_kunpackb(__A, __B);
  // CIR: [[MASK_A:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 16>), !cir.vector<!cir.bool x 16>
  // CIR: [[EXTRACT_A:%.*]] = cir.vec.shuffle([[MASK_A]], {{.*}}) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]
  // CIR: [[MASK_B:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 16>), !cir.vector<!cir.bool x 16>
  // CIR: [[EXTRACT_B:%.*]] = cir.vec.shuffle([[MASK_B]], {{.*}}) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]
  // CIR: [[CONCAT:%.*]] = cir.vec.shuffle([[EXTRACT_B]], [[EXTRACT_A]]) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i]
  // CIR: {{%.*}} = cir.cast(bitcast, [[CONCAT]] : !cir.vector<!cir.bool x 16>), !cir.int<u, 16>

  // LLVM: [[A_BITCAST:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[A_SHUFFLE:%.*]] = shufflevector <16 x i1> [[A_BITCAST]], <16 x i1> [[A_BITCAST]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: [[B_BITCAST:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[B_SHUFFLE:%.*]] = shufflevector <16 x i1> [[B_BITCAST]], <16 x i1> [[B_BITCAST]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: [[CONCAT:%.*]] = shufflevector <8 x i1> [[B_SHUFFLE]], <8 x i1> [[A_SHUFFLE]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: bitcast <16 x i1> [[CONCAT]] to i16

  // OGCG: [[A_BITCAST:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: [[A_SHUFFLE:%.*]] = shufflevector <16 x i1> [[A_BITCAST]], <16 x i1> [[A_BITCAST]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: [[B_BITCAST:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: [[B_SHUFFLE:%.*]] = shufflevector <16 x i1> [[B_BITCAST]], <16 x i1> [[B_BITCAST]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: [[CONCAT:%.*]] = shufflevector <8 x i1> [[B_SHUFFLE]], <8 x i1> [[A_SHUFFLE]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: bitcast <16 x i1> [[CONCAT]] to i16
}

__mmask32 test_mm512_kunpackw(__mmask32 __A, __mmask32 __B) {
  // CIR-LABEL: test_mm512_kunpackw
  // LLVM-LABEL: test_mm512_kunpackw
  // OGCG-LABEL: test_mm512_kunpackw
  return _mm512_kunpackw(__A, __B);
  // CIR: [[MASK_A:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 32>), !cir.vector<!cir.bool x 32>
  // CIR: [[EXTRACT_A:%.*]] = cir.vec.shuffle([[MASK_A]], {{.*}})
  // CIR: [[MASK_B:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 32>), !cir.vector<!cir.bool x 32>
  // CIR: [[EXTRACT_B:%.*]] = cir.vec.shuffle([[MASK_B]], {{.*}})
  // CIR: [[CONCAT:%.*]] = cir.vec.shuffle([[EXTRACT_B]], [[EXTRACT_A]])
  // CIR: {{%.*}} = cir.cast(bitcast, [[CONCAT]] : !cir.vector<!cir.bool x 32>), !cir.int<u, 32>

  // LLVM: [[A_BITCAST:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[A_SHUFFLE:%.*]] = shufflevector <32 x i1> [[A_BITCAST]], <32 x i1> [[A_BITCAST]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: [[B_BITCAST:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[B_SHUFFLE:%.*]] = shufflevector <32 x i1> [[B_BITCAST]], <32 x i1> [[B_BITCAST]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: [[CONCAT:%.*]] = shufflevector <16 x i1> [[B_SHUFFLE]], <16 x i1> [[A_SHUFFLE]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: bitcast <32 x i1> [[CONCAT]] to i32

  // OGCG: [[A_BITCAST:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[A_SHUFFLE:%.*]] = shufflevector <32 x i1> [[A_BITCAST]], <32 x i1> [[A_BITCAST]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: [[B_BITCAST:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[B_SHUFFLE:%.*]] = shufflevector <32 x i1> [[B_BITCAST]], <32 x i1> [[B_BITCAST]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: [[CONCAT:%.*]] = shufflevector <16 x i1> [[B_SHUFFLE]], <16 x i1> [[A_SHUFFLE]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: bitcast <32 x i1> [[CONCAT]] to i32
}

__mmask64 test_mm512_kunpackd(__mmask64 __A, __mmask64 __B) {
  // CIR-LABEL: test_mm512_kunpackd
  // LLVM-LABEL: test_mm512_kunpackd
  // OGCG-LABEL: test_mm512_kunpackd
  return _mm512_kunpackd(__A, __B);
  // CIR: [[MASK_A:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 64>), !cir.vector<!cir.bool x 64>
  // CIR: [[EXTRACT_A:%.*]] = cir.vec.shuffle([[MASK_A]], {{.*}})
  // CIR: [[MASK_B:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.int<u, 64>), !cir.vector<!cir.bool x 64>
  // CIR: [[EXTRACT_B:%.*]] = cir.vec.shuffle([[MASK_B]], {{.*}})
  // CIR: [[CONCAT:%.*]] = cir.vec.shuffle([[EXTRACT_B]], [[EXTRACT_A]])
  // CIR: {{%.*}} = cir.cast(bitcast, [[CONCAT]] : !cir.vector<!cir.bool x 64>), !cir.int<u, 64>

  // LLVM: [[A_BITCAST:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[A_SHUFFLE:%.*]] = shufflevector <64 x i1> [[A_BITCAST]], <64 x i1> [[A_BITCAST]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: [[B_BITCAST:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[B_SHUFFLE:%.*]] = shufflevector <64 x i1> [[B_BITCAST]], <64 x i1> [[B_BITCAST]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: [[CONCAT:%.*]] = shufflevector <32 x i1> [[B_SHUFFLE]], <32 x i1> [[A_SHUFFLE]], <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  // LLVM: bitcast <64 x i1> [[CONCAT]] to i64

  // OGCG: [[A_BITCAST:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[A_SHUFFLE:%.*]] = shufflevector <64 x i1> [[A_BITCAST]], <64 x i1> [[A_BITCAST]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: [[B_BITCAST:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[B_SHUFFLE:%.*]] = shufflevector <64 x i1> [[B_BITCAST]], <64 x i1> [[B_BITCAST]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: [[CONCAT:%.*]] = shufflevector <32 x i1> [[B_SHUFFLE]], <32 x i1> [[A_SHUFFLE]], <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  // OGCG: bitcast <64 x i1> [[CONCAT]] to i64
}


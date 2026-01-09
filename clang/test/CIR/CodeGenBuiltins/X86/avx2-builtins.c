// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/avx2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m256i test0_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CIR-LABEL: test0_mm256_inserti128_si256
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s64i>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s64i>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s64i>

  // LLVM-LABEL: test0_mm256_inserti128_si256
  // LLVM: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>

  // OGCG-LABEL: test0_mm256_inserti128_si256
  // OGCG: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_inserti128_si256(a, b, 0);
}

__m256i test1_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CIR-LABEL: test1_mm256_inserti128_si256
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s64i>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s64i>

  // LLVM-LABEL: test1_mm256_inserti128_si256
  // LLVM: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>

  // OGCG-LABEL: test1_mm256_inserti128_si256
  // OGCG: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti128_si256(a, b, 1);
}

__m256i test_mm256_shufflelo_epi16(__m256i a) {
  // CIR-LABEL: _mm256_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s16i>) [#cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<11> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<9> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<16 x !s16i>

  // LLVM-LABEL: test_mm256_shufflelo_epi16
  // LLVM: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 3, i32 0, i32 1, i32 1, i32 4, i32 5, i32 6, i32 7, i32 11, i32 8, i32 9, i32 9, i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm256_shufflelo_epi16
  // OGCG: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 3, i32 0, i32 1, i32 1, i32 4, i32 5, i32 6, i32 7, i32 11, i32 8, i32 9, i32 9, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shufflelo_epi16(a, 83);
}

__m256i test_mm256_shufflehi_epi16(__m256i a) {
  // CIR-LABEL: _mm256_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i, #cir.int<6> : !s32i, #cir.int<6> : !s32i, #cir.int<5> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<15> : !s32i, #cir.int<14> : !s32i, #cir.int<14> : !s32i, #cir.int<13> : !s32i] : !cir.vector<16 x !s16i>

  // LLVM-LABEL: test_mm256_shufflehi_epi16
  // LLVM: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 6, i32 5, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 14, i32 13>

  // OGCG-LABEL: test_mm256_shufflehi_epi16
  // OGCG: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 6, i32 5, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 14, i32 13>
  return _mm256_shufflehi_epi16(a, 107);
}

__m256i test_mm256_mul_epu32(__m256i a, __m256i b) {
  // CIR-LABEL: _mm256_mul_epu32
  // CIR: [[BC_A:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
  // CIR: [[BC_B:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
  // CIR: [[MASK_SCALAR:%.*]] = cir.const #cir.int<4294967295> : !s64i
  // CIR: [[MASK_VEC:%.*]] = cir.vec.splat [[MASK_SCALAR]] : !s64i, !cir.vector<4 x !s64i>
  // CIR: [[AND_A:%.*]] = cir.binop(and, [[BC_A]], [[MASK_VEC]])
  // CIR: [[AND_B:%.*]] = cir.binop(and, [[BC_B]], [[MASK_VEC]])
  // CIR: [[MUL:%.*]]   = cir.binop(mul, [[AND_A]], [[AND_B]])

  // LLVM-LABEL: _mm256_mul_epu32
  // LLVM: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: mul <4 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm256_mul_epu32
  // OGCG: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: and <4 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: mul <4 x i64> %{{.*}}, %{{.*}}

  return _mm256_mul_epu32(a, b);
}

__m256i test_mm256_mul_epi32(__m256i a, __m256i b) {
  // CIR-LABEL: _mm256_mul_epi32
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<4 x !s64i>
  // CIR: [[SHL_A:%.*]]  = cir.shift(left, [[A64]] : !cir.vector<4 x !s64i>, [[SV]] : !cir.vector<4 x !s64i>)
  // CIR: [[ASHR_A:%.*]] = cir.shift(right, [[SHL_A]] : !cir.vector<4 x !s64i>, [[SV]] : !cir.vector<4 x !s64i>)
  // CIR: [[SHL_B:%.*]]  = cir.shift(left, [[B64]] : !cir.vector<4 x !s64i>, [[SV]] : !cir.vector<4 x !s64i>)
  // CIR: [[ASHR_B:%.*]] = cir.shift(right, [[SHL_B]] : !cir.vector<4 x !s64i>, [[SV]] : !cir.vector<4 x !s64i>)
  // CIR: [[MUL:%.*]]    = cir.binop(mul, [[ASHR_A]], [[ASHR_B]])

  // LLVM-LABEL: _mm256_mul_epi32
  // LLVM: shl <4 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // LLVM: shl <4 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // LLVM: mul <4 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm256_mul_epi32
  // OGCG: shl <4 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // OGCG: shl <4 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <4 x i64> %{{.*}}, splat (i64 32)
  // OGCG: mul <4 x i64> %{{.*}}, %{{.*}}

  return _mm256_mul_epi32(a, b);
}

__m256i test_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CIR-LABEL: _mm256_alignr_epi8
  // CIR: {{.*}} = cir.vec.shuffle(%{{.*}}, {{.*}} : !cir.vector<32 x {{![us]8i}}>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i] : !cir.vector<32 x {{![us]8i}}>

  // LLVM-LABEL: test_mm256_alignr_epi8
  // LLVM: shufflevector <32 x i8> {{.*}}, <32 x i8> {{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>

  // OGCG-LABEL: test_mm256_alignr_epi8
  // OGCG: shufflevector <32 x i8> {{.*}}, <32 x i8> {{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  return _mm256_alignr_epi8(a, b, 2);
}

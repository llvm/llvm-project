// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512bw  -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__mmask16 test_mm_cmp_epi8_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_cmp_epi8_mask
  // LLVM: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epi8_mask
  // OGCG: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epi8_mask(__a, __b, 0);
}

__mmask16 test_mm_cmp_epi8_mask_imm3(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi8_mask_imm3
  // CIR: cir.const #cir.zero : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_cmp_epi8_mask_imm3
  // LLVM: store i16 0, ptr %{{.*}}, align 2
  // LLVM: load i16, ptr %{{.*}}, align 2
  // LLVM: ret i16 %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epi8_mask_imm3
  // OGCG: ret i16 0
  return (__mmask16)_mm_cmp_epi8_mask(__a, __b, 3);
}

__mmask16 test_mm_cmp_epi8_mask_imm7(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi8_mask_imm7
  // CIR: cir.const #cir.int<-1> : !cir.int<s, 1>
  // CIR: cir.vec.splat {{%.*}} : !cir.int<s, 1>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_cmp_epi8_mask_imm7
  // LLVM: store i16 -1, ptr %{{.*}}
  // LLVM: load i16, ptr %{{.*}}
  // LLVM: ret i16 %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epi8_mask_imm7
  // OGCG: ret i16 -1
  return (__mmask16)_mm_cmp_epi8_mask(__a, __b, 7);
}

__mmask16 test_mm_mask_cmp_epi8_mask(__mmask16 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_mask_cmp_epi8_mask
  // LLVM: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_mask_cmp_epi8_mask
  // OGCG: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epi8_mask(__m, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epi8_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s8i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm256_cmp_epi8_mask
  // LLVM: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epi8_mask
  // OGCG: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epi8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epi8_mask(__mmask32 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s8i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm256_mask_cmp_epi8_mask
  // LLVM: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <32 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epi8_mask
  // OGCG: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epi8_mask(__m, __a, __b, 0);
}

__mmask64 test_mm512_cmp_epi8_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<64 x !s8i>, !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // LLVM-LABEL: test_mm512_cmp_epi8_mask
  // LLVM: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epi8_mask
  // OGCG: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmp_epi8_mask(__a, __b, 0);
}

__mmask64 test_mm512_mask_cmp_epi8_mask(__mmask64 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epi8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<64 x !s8i>, !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // LLVM-LABEL: test_mm512_mask_cmp_epi8_mask
  // LLVM: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <64 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epi8_mask
  // OGCG: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmp_epi8_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi16_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epi16_mask
  // LLVM: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epi16_mask
  // OGCG: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi16_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epi16_mask
  // LLVM: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <8 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_mask_cmp_epi16_mask
  // OGCG: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi16_mask(__m, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epi16_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s16i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm256_cmp_epi16_mask
  // LLVM: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epi16_mask
  // OGCG: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epi16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epi16_mask(__mmask16 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s16i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm256_mask_cmp_epi16_mask
  // LLVM: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epi16_mask
  // OGCG: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epi16_mask(__m, __a, __b, 0);
}

__mmask32 test_mm512_cmp_epi16_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s16i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm512_cmp_epi16_mask
  // LLVM: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epi16_mask
  // OGCG: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmp_epi16_mask(__a, __b, 0);
}

__mmask32 test_mm512_mask_cmp_epi16_mask(__mmask32 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epi16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s16i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm512_mask_cmp_epi16_mask
  // LLVM: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <32 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epi16_mask
  // OGCG: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmp_epi16_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi32_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epi32_mask
  // LLVM: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm_cmp_epi32_mask
  // OGCG: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi32_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epi32_mask
  // LLVM: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: and <4 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm_mask_cmp_epi32_mask
  // OGCG: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: and <4 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_mask_cmp_epi32_mask(__m, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epi32_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s32i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_cmp_epi32_mask
  // LLVM: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epi32_mask
  // OGCG: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epi32_mask(__mmask8 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s32i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_mask_cmp_epi32_mask
  // LLVM: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // LLVM: and <8 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epi32_mask
  // OGCG: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__m, __a, __b, 0);
}

__mmask16 test_mm512_cmp_epi32_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s32i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm512_cmp_epi32_mask
  // LLVM: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epi32_mask
  // OGCG: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmp_epi32_mask(__a, __b, 0);
}

__mmask16 test_mm512_mask_cmp_epi32_mask(__mmask16 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epi32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s32i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm512_mask_cmp_epi32_mask
  // LLVM: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epi32_mask
  // OGCG: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmp_epi32_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi64_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epi64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epi64_mask
  // LLVM: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG-LABEL: test_mm_cmp_epi64_mask
  // OGCG: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi64_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epi64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epi64_mask
  // LLVM: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // LLVM: and <2 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG-LABEL: test_mm_mask_cmp_epi64_mask
  // OGCG: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // OGCG: and <2 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_mask_cmp_epi64_mask(__m, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epi64_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epi64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s64i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_cmp_epi64_mask
  // LLVM: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm256_cmp_epi64_mask
  // OGCG: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epi64_mask(__mmask8 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epi64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s64i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_mask_cmp_epi64_mask
  // LLVM: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: and <4 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm256_mask_cmp_epi64_mask
  // OGCG: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: and <4 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__m, __a, __b, 0);
}

__mmask16 test_mm_cmp_epu8_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_cmp_epu8_mask
  // LLVM: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epu8_mask
  // OGCG: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epu8_mask(__a, __b, 0);
}

__mmask16 test_mm_mask_cmp_epu8_mask(__mmask16 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm_mask_cmp_epu8_mask
  // LLVM: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_mask_cmp_epu8_mask
  // OGCG: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epu8_mask(__m, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epu8_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s8i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm256_cmp_epu8_mask
  // LLVM: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epu8_mask
  // OGCG: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epu8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epu8_mask(__mmask32 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s8i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm256_mask_cmp_epu8_mask
  // LLVM: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <32 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epu8_mask
  // OGCG: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epu8_mask(__m, __a, __b, 0);
}

__mmask64 test_mm512_cmp_epu8_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<64 x !s8i>, !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // LLVM-LABEL: test_mm512_cmp_epu8_mask
  // LLVM: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epu8_mask
  // OGCG: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmp_epu8_mask(__a, __b, 0);
}

__mmask64 test_mm512_mask_cmp_epu8_mask(__mmask64 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epu8_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<64 x !s8i>, !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // LLVM-LABEL: test_mm512_mask_cmp_epu8_mask
  // LLVM: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // LLVM: and <64 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epu8_mask
  // OGCG: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // OGCG: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmp_epu8_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu16_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epu16_mask
  // LLVM: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_cmp_epu16_mask
  // OGCG: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu16_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epu16_mask
  // LLVM: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <8 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm_mask_cmp_epu16_mask
  // OGCG: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu16_mask(__m, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epu16_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s16i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm256_cmp_epu16_mask
  // LLVM: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epu16_mask
  // OGCG: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epu16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epu16_mask(__mmask16 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s16i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm256_mask_cmp_epu16_mask
  // LLVM: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epu16_mask
  // OGCG: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epu16_mask(__m, __a, __b, 0);
}

__mmask32 test_mm512_cmp_epu16_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s16i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm512_cmp_epu16_mask
  // LLVM: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epu16_mask
  // OGCG: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmp_epu16_mask(__a, __b, 0);
}

__mmask32 test_mm512_mask_cmp_epu16_mask(__mmask32 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epu16_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<32 x !s16i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // LLVM-LABEL: test_mm512_mask_cmp_epu16_mask
  // LLVM: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // LLVM: and <32 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epu16_mask
  // OGCG: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // OGCG: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmp_epu16_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epu32_mask
  // LLVM: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm_cmp_epu32_mask
  // OGCG: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epu32_mask
  // LLVM: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: and <4 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm_mask_cmp_epu32_mask
  // OGCG: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: and <4 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_mask_cmp_epu32_mask(__m, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s32i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_cmp_epu32_mask
  // LLVM: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_cmp_epu32_mask
  // OGCG: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s32i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_mask_cmp_epu32_mask
  // LLVM: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // LLVM: and <8 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm256_mask_cmp_epu32_mask
  // OGCG: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__m, __a, __b, 0);
}

__mmask16 test_mm512_cmp_epu32_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s32i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm512_cmp_epu32_mask
  // LLVM: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epu32_mask
  // OGCG: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmp_epu32_mask(__a, __b, 0);
}

__mmask16 test_mm512_mask_cmp_epu32_mask(__mmask16 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epu32_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<16 x !s32i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // LLVM-LABEL: test_mm512_mask_cmp_epu32_mask
  // LLVM: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // LLVM: and <16 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epu32_mask
  // OGCG: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // OGCG: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmp_epu32_mask(__m, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_cmp_epu64_mask
  // LLVM: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG-LABEL: test_mm_cmp_epu64_mask
  // OGCG: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __m, __m128i __a, __m128i __b) {
  // CIR-LABEL: test_mm_mask_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<2 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm_mask_cmp_epu64_mask
  // LLVM: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // LLVM: and <2 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG-LABEL: test_mm_mask_cmp_epu64_mask
  // OGCG: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // OGCG: and <2 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_mask_cmp_epu64_mask(__m, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s64i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_cmp_epu64_mask
  // LLVM: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm256_cmp_epu64_mask
  // OGCG: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __m, __m256i __a, __m256i __b) {
  // CIR-LABEL: test_mm256_mask_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<4 x !s64i>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.const #cir.zero : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.int<s, 1>>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm256_mask_cmp_epu64_mask
  // LLVM: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // LLVM: bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: and <4 x i1> %{{.*}}, %{{.*}}
  // LLVM: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_mm256_mask_cmp_epu64_mask
  // OGCG: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // OGCG: bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: and <4 x i1> %{{.*}}, %{{.*}}
  // OGCG: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__m, __a, __b, 0);
}

__mmask8 test_mm512_cmp_epu64_mask(__m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s64i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm512_cmp_epu64_mask
  // LLVM: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_cmp_epu64_mask
  // OGCG: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm512_mask_cmp_epu64_mask(__mmask8 __m, __m512i __a, __m512i __b) {
  // CIR-LABEL: test_mm512_mask_cmp_epu64_mask
  // CIR: cir.vec.cmp(eq, {{%.*}}, {{%.*}}) : !cir.vector<8 x !s64i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{%.*}}, {{%.*}}) : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{%.*}} : !cir.vector<8 x !cir.int<s, 1>> -> !u8i
  // LLVM-LABEL: test_mm512_mask_cmp_epu64_mask
  // LLVM: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // LLVM: and <8 x i1> %{{.*}}, %{{.*}}
  // OGCG-LABEL: test_mm512_mask_cmp_epu64_mask
  // OGCG: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmp_epu64_mask(__m, __a, __b, 0);
}

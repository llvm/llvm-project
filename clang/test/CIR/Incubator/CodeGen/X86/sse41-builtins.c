// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/sse41-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

int test_mm_extract_epi8(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi8
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s8i x 16>
  // CIR-CHECK %{{.*}} = cir.cast integral %{{.*}} : !u8i -> !s32i

  // LLVM-CHECK-LABEL: test_mm_extract_epi8
  // LLVM-CHECK: extractelement <16 x i8> %{{.*}}, {{i32|i64}} 1
  // LLVM-CHECK: zext i8 %{{.*}} to i32
  return _mm_extract_epi8(x, 1);
}

int test_mm_extract_epi32(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi32
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 4>

  // LLVM-CHECK-LABEL: test_mm_extract_epi32
  // LLVM-CHECK: extractelement <4 x i32> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi32(x, 1);
}

long long test_mm_extract_epi64(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi64
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s64i x 2>

  // LLVM-CHECK-LABEL: test_mm_extract_epi64
  // LLVM-CHECK: extractelement <2 x i64> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi64(x, 1);
}

int test_mm_extract_ps(__m128 x) {
  // CIR-CHECK-LABEL: test_mm_extract_ps
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!cir.float x 4>

  // LLVM-CHECK-LABEL: test_mm_extract_ps
  // LLVM-CHECK: extractelement <4 x float> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_ps(x, 1);
}

__m128i test_mm_insert_epi8(__m128i x, char b) {

  // CIR-CHECK-LABEL: test_mm_insert_epi8
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<{{!s8i|!u8i}} x 16>

  // LLVM-CHECK-LABEL: test_mm_insert_epi8 
  // LLVM-CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi8(x, b, 1);
}

__m128i test_mm_insert_epi32(__m128i x, int b) {

  // CIR-CHECK-LABEL: test_mm_insert_epi32
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 4>

  // LLVM-CHECK-LABEL: test_mm_insert_epi32
  // LLVM-CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi32(x, b, 1);
}

#ifdef __x86_64__
__m128i test_mm_insert_epi64(__m128i x, long long b) {

  // CIR-X64-LABEL: test_mm_insert_epi64
  // CIR-X64: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s64i x 2>

  // LLVM-X64-LABEL: test_mm_insert_epi64
  // LLVM-X64: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi64(x, b, 1);
}
#endif

__m128i test_mm_blend_epi16(__m128i V1, __m128i V2) {
  // CIR-LABEL: test_mm_blend_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s16i x 8>) [#cir.int<0> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_mm_blend_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>

  // OGCG-LABEL: test_mm_blend_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}

__m128d test_mm_blend_pd(__m128d V1, __m128d V2) {
  // CIR-LABEL: test_mm_blend_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s64i x 2>) [#cir.int<0> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s64i x 2>

  // LLVM-LABEL: test_mm_blend_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>

  // OGCG-LABEL: test_mm_blend_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_blend_pd(V1, V2, 2);
}

__m128 test_mm_blend_ps(__m128 V1, __m128 V2) {
  // CIR-LABEL: test_mm_blend_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 4>

  // LLVM-LABEL: test_mm_blend_ps
  // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>

  // OGCG-LABEL: test_mm_blend_ps
  // OGCG: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 6);
}

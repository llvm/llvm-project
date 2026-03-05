// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/avx-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m256 test_mm256_undefined_ps(void) {
  // CIR-X64-LABEL: _mm256_undefined_ps
  // CIR-X64: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 4>
  // CIR-X64: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 4> -> !cir.vector<!cir.float x 8>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<!cir.float x 8>

  // LLVM-X64-LABEL: test_mm256_undefined_ps
  // LLVM-X64: store <8 x float> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM-X64: %{{.*}} = load <8 x float>, ptr %[[A]], align 32
  // LLVM-X64: ret <8 x float> %{{.*}}

  return _mm256_undefined_ps();
}

__m256d test_mm256_undefined_pd(void) {
  // CIR-X64-LABEL: _mm256_undefined_pd
  // CIR-X64: %{{.*}} = cir.const #cir.zero : !cir.vector<!cir.double x 4>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<!cir.double x 4>

  // LLVM-X64-LABEL: test_mm256_undefined_pd
  // LLVM-X64: store <4 x double> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM-X64: %{{.*}} = load <4 x double>, ptr %[[A]], align 32
  // LLVM-X64: ret <4 x double> %{{.*}}

  return _mm256_undefined_pd();
}

__m256i test_mm256_undefined_si256(void) {
  // CIR-X64-LABEL: _mm256_undefined_si256
  // CIR-X64: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 4>
  // CIR-X64: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 4> -> !cir.vector<!s64i x 4>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<!s64i x 4>
  
  // LLVM-X64-LABEL: test_mm256_undefined_si256
  // LLVM-X64: store <4 x i64> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM-X64: %{{.*}} = load <4 x i64>, ptr %[[A]], align 32
  // LLVM-X64: ret <4 x i64> %{{.*}}
  return _mm256_undefined_si256();
}

int test_mm256_extract_epi8(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi8
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s8i x 32>
  // CIR-CHECK %{{.*}} = cir.cast integral %{{.*}} : !u8i -> !s32i

  // LLVM-CHECK-LABEL: test_mm256_extract_epi8
  // LLVM-CHECK: extractelement <32 x i8> %{{.*}}, {{i32|i64}} 31
  // LLVM-CHECK: zext i8 %{{.*}} to i32
  return _mm256_extract_epi8(A, 31);
}

int test_mm256_extract_epi16(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi16
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 16>
  // CIR-CHECK %{{.*}} = cir.cast integral %{{.*}} : !u16i -> !s32i

  // LLVM-CHECK-LABEL: test_mm256_extract_epi16
  // LLVM-CHECK: extractelement <16 x i16> %{{.*}}, {{i32|i64}} 15
  // LLVM-CHECK: zext i16 %{{.*}} to i32
  return _mm256_extract_epi16(A, 15);
}

int test_mm256_extract_epi32(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi32
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 8>

  // LLVM-CHECK-LABEL: test_mm256_extract_epi32
  // LLVM-CHECK: extractelement <8 x i32> %{{.*}}, {{i32|i64}} 7
  return _mm256_extract_epi32(A, 7);
}

#if __x86_64__
long long test_mm256_extract_epi64(__m256i A) {
  // CIR-X64-LABEL: test_mm256_extract_epi64
  // LLVM-X64-LABEL: test_mm256_extract_epi64
  return _mm256_extract_epi64(A, 3);
}
#endif

__m256i test_mm256_insert_epi8(__m256i x, char b) {

  // CIR-CHECK-LABEL: test_mm256_insert_epi8
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<{{!s8i|!u8i}} x 32>

  // LLVM-CHECK-LABEL: test_mm256_insert_epi8
  // LLVM-CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 14
  return _mm256_insert_epi8(x, b, 14);
}

__m256i test_mm256_insert_epi16(__m256i x, int b) {

  // CIR-CHECK-LABEL: test_mm256_insert_epi16
  // CIR-CHECK: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 16>

  // LLVM-CHECK-LABEL: test_mm256_insert_epi16
  // LLVM-CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, {{i32|i64}} 4
  return _mm256_insert_epi16(x, b, 4);
}

__m256i test_mm256_insert_epi32(__m256i x, int b) {

  // CIR-CHECK-LABEL: test_mm256_insert_epi32
  // CIR-CHECK: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 8>

  // LLVM-CHECK-LABEL: test_mm256_insert_epi32
  // LLVM-CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 5
  return _mm256_insert_epi32(x, b, 5);
}

#ifdef __x86_64__
__m256i test_mm256_insert_epi64(__m256i x, long long b) {

  // CIR-X64-LABEL: test_mm256_insert_epi64
  // CIR-X64: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s64i x 4>

  // LLVM-X64-LABEL: test_mm256_insert_epi64
  // LLVM-X64: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 2
  return _mm256_insert_epi64(x, b, 2);
}
#endif

__m256d test_mm256_blend_pd(__m256d A, __m256d B) {
  // CIR-LABEL: test_mm256_blend_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 4>) [#cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.double x 4>

  // LLVM-LABEL: test_mm256_blend_pd
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>

  // OGCG-LABEL: test_mm256_blend_pd
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm256_blend_pd(A, B, 0x05);
}

__m256 test_mm256_blend_ps(__m256 A, __m256 B) {
  // CIR-LABEL: test_mm256_blend_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 8>) [#cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!cir.float x 8>

  // LLVM-LABEL: test_mm256_blend_ps
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>

  // OGCG-LABEL: test_mm256_blend_ps
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_ps(A, B, 0x35);
}

__m256d test_mm256_insertf128_pd(__m256d A, __m128d B) {
  // CIR-LABEL: test_mm256_insertf128_pd
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.double x 4>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 4>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.double x 4>

  // LLVM-LABEL: test_mm256_insertf128_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_insertf128_pd(A, B, 0);
}

__m256 test_mm256_insertf128_ps(__m256 A, __m128 B) {
  // CIR-LABEL: test_mm256_insertf128_ps
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.float x 8>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<!cir.float x 8>

  // LLVM-LABEL: test_mm256_insertf128_ps
  // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf128_ps(A, B, 1);
}

__m256i test_mm256_insertf128_si256(__m256i A, __m128i B) {
  // CIR-LABEL: test_mm256_insertf128_si256
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 8>
  // %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 8>) [#cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]

  // LLVM-LABEL: test_mm256_insertf128_si256
  // LLVM: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  return _mm256_insertf128_si256(A, B, 0);
}

__m256d test_mm256_shuffle_pd(__m256d A, __m256d B) {
  // CIR-LABEL: test_mm256_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 4>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<2> : !s32i, #cir.int<6> : !s32i] : !cir.vector<!cir.double x 4>

  // CHECK-LABEL: test_mm256_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>

  // OGCG-LABEL: test_mm256_shuffle_pd
  // OGCG: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_shuffle_pd(A, B, 0);
}

__m256 test_mm256_shuffle_ps(__m256 A, __m256 B) {
  // CIR-LABEL: test_mm256_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 8>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<8> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<12> : !s32i] : !cir.vector<!cir.float x 8>

  // CHECK-LABEL: test_mm256_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 8, i32 8, i32 4, i32 4, i32 12, i32 12>

  // OGCG-LABEL: test_mm256_shuffle_ps
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 8, i32 8, i32 4, i32 4, i32 12, i32 12>
  return _mm256_shuffle_ps(A, B, 0);
}


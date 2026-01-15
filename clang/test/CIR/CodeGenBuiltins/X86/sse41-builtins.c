// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m128i test_mm_mul_epi32(__m128i x, __m128i y) {
  // CIR-LABEL: _mm_mul_epi32
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<2 x !s64i>
  // CIR: [[SHL_A:%.*]]  = cir.shift(left, [[A64]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[ASHR_A:%.*]] = cir.shift(right, [[SHL_A]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[SHL_B:%.*]]  = cir.shift(left, [[B64]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[ASHR_B:%.*]] = cir.shift(right, [[SHL_B]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[MUL:%.*]]    = cir.binop(mul, [[ASHR_A]], [[ASHR_B]])

  // LLVM-LABEL: _mm_mul_epi32
  // LLVM: shl <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: shl <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: mul <2 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm_mul_epi32
  // OGCG: shl <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: shl <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: mul <2 x i64> %{{.*}}, %{{.*}}

  return _mm_mul_epi32(x, y);
}

__m128i test_mm_blend_epi16(__m128i V1, __m128i V2) {
  // CIR-LABEL: test_mm_blend_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s16i>

  // LLVM-LABEL: test_mm_blend_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>

  // OGCG-LABEL: test_mm_blend_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}

__m128d test_mm_blend_pd(__m128d V1, __m128d V2) {
  // CIR-LABEL: test_mm_blend_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_blend_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>

  // OGCG-LABEL: test_mm_blend_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_blend_pd(V1, V2, 2);
}

__m128 test_mm_blend_ps(__m128 V1, __m128 V2) {
  // CIR-LABEL: test_mm_blend_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_blend_ps
  // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>

  // OGCG-LABEL: test_mm_blend_ps
  // OGCG: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 6);
}

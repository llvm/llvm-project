// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
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

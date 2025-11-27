// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx2 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx2 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m256i test_mm256_mul_epu32(__m256i a, __m256i b) {
// CIR-LABEL: _mm256_mul_epu32
// CIR: [[MASK_SCALAR:%.*]] = cir.const #cir.int<4294967295> : !s64i
// CIR: [[MASK_VEC:%.*]] = cir.vec.splat [[MASK_SCALAR]] : !s64i, !cir.vector<4 x !s64i>
// CIR: [[BC_A:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
// CIR: [[BC_B:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
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
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<4 x !s64i>
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !s64i>
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

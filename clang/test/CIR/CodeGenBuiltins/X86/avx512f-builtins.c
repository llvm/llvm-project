// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m512 test_mm512_undefined(void) {
  // CIR-LABEL: _mm512_undefined
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps(void) {
  // CIR-LABEL: _mm512_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined_ps
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_ps
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd(void) {
  // CIR-LABEL: _mm512_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.double>

  // LLVM-LABEL: test_mm512_undefined_pd
  // LLVM: store <8 x double> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x double>, ptr %[[A]], align 64
  // LLVM: ret <8 x double> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_pd
  // OGCG: ret <8 x double> zeroinitializer
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32(void) {
  // CIR-LABEL: _mm512_undefined_epi32
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<8 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !s64i>

  // LLVM-LABEL: test_mm512_undefined_epi32
  // LLVM: store <8 x i64> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x i64>, ptr %[[A]], align 64
  // LLVM: ret <8 x i64> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_epi32
  // OGCG: ret <8 x i64> zeroinitializer
  return _mm512_undefined_epi32();
}

__m512i test_mm512_mul_epi32(__m512i __A, __m512i __B) {
  // CIR-LABEL: _mm512_mul_epi32
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<8 x !s64i>
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: [[SHL_A:%.*]]  = cir.shift(left, [[A64]] : !cir.vector<8 x !s64i>, [[SV]] : !cir.vector<8 x !s64i>)
  // CIR: [[ASHR_A:%.*]] = cir.shift(right, [[SHL_A]] : !cir.vector<8 x !s64i>, [[SV]] : !cir.vector<8 x !s64i>)
  // CIR: [[SHL_B:%.*]]  = cir.shift(left, [[B64]] : !cir.vector<8 x !s64i>, [[SV]] : !cir.vector<8 x !s64i>)
  // CIR: [[ASHR_B:%.*]] = cir.shift(right, [[SHL_B]] : !cir.vector<8 x !s64i>, [[SV]] : !cir.vector<8 x !s64i>)
  // CIR: [[MUL:%.*]]    = cir.binop(mul, [[ASHR_A]], [[ASHR_B]])

  // LLVM-LABEL: _mm512_mul_epi32
  // LLVM: shl <8 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <8 x i64> %{{.*}}, splat (i64 32)
  // LLVM: shl <8 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <8 x i64> %{{.*}}, splat (i64 32)
  // LLVM: mul <8 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm512_mul_epi32
  // OGCG: shl <8 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <8 x i64> %{{.*}}, splat (i64 32)
  // OGCG: shl <8 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <8 x i64> %{{.*}}, splat (i64 32)
  // OGCG: mul <8 x i64> %{{.*}}, %{{.*}}

  return _mm512_mul_epi32(__A, __B);
}


__m512i test_mm512_mul_epu32(__m512i __A, __m512i __B) {
  // CIR-LABEL: _mm512_mul_epu32
  // CIR: [[MASK_SCALAR:%.*]] = cir.const #cir.int<4294967295> : !s64i
  // CIR: [[MASK_VEC:%.*]] = cir.vec.splat [[MASK_SCALAR]] : !s64i, !cir.vector<8 x !s64i>
  // CIR: [[BC_A:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !s64i>
  // CIR: [[BC_B:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !s64i>
  // CIR: [[AND_A:%.*]] = cir.binop(and, [[BC_A]], [[MASK_VEC]])
  // CIR: [[AND_B:%.*]] = cir.binop(and, [[BC_B]], [[MASK_VEC]])
  // CIR: [[MUL:%.*]]   = cir.binop(mul, [[AND_A]], [[AND_B]])

  // LLVM-LABEL: _mm512_mul_epu32
  // LLVM: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: mul <8 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm512_mul_epu32
  // OGCG: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: mul <8 x i64> %{{.*}}, %{{.*}}

  return _mm512_mul_epu32(__A, __B);
}

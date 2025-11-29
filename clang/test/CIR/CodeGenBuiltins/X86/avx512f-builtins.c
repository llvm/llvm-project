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

__m512d test_mm512_shuffle_pd(__m512d __M, __m512d __V) {
  // CIR-LABEL: test_mm512_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<3> : !s32i, #cir.int<10> : !s32i, #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !cir.double>

  // LLVM-LABEL: test_mm512_shuffle_pd
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>

  // OGCG-LABEL: test_mm512_shuffle_pd
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_shuffle_pd(__M, __V, 4);
}

__m512 test_mm512_shuffle_ps(__m512 __M, __m512 __V) {
  // CIR-LABEL: test_mm512_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<16> : !s32i, #cir.int<16> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<20> : !s32i, #cir.int<20> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<24> : !s32i, #cir.int<24> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<28> : !s32i, #cir.int<28> : !s32i] : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_shuffle_ps
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>

  // OGCG-LABEL: test_mm512_shuffle_ps
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>
  return _mm512_shuffle_ps(__M, __V, 4);
}

__m512 test_mm512_permute_ps(__m512 A) {
    // CIR-LABEL: test_mm512_permute_ps
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !cir.float>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i] : !cir.vector<16 x !cir.float>

    // LLVM-LABEL: test_mm512_permute_ps
    // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <16 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9, i32 14, i32 15, i32 12, i32 13>

    // OGCG-LABEL: test_mm512_permute_ps
    // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <16 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9, i32 14, i32 15, i32 12, i32 13>
    return _mm512_permute_ps(A, 0x4E);
}

__m512d test_mm512_permute_pd(__m512d A) {
    // CIR-LABEL: test_mm512_permute_pd
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<7> : !s32i, #cir.int<6> : !s32i] : !cir.vector<8 x !cir.double>

    // LLVM-LABEL: test_mm512_permute_pd
    // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>

    // OGCG-LABEL: test_mm512_permute_pd
    // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
    return _mm512_permute_pd(A, 0x55);
}
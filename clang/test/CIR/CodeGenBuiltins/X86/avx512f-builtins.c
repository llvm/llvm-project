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

__m512d test_mm512_insertf64x4(__m512d __A, __m256d __B) {
  // CIR-LABEL: test_mm512_insertf64x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !cir.double>

  // LLVM-LABEL: test_mm512_insertf64x4
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>

  // OGCG-LABEL: test_mm512_insertf64x4
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_insertf64x4(__A, __B, 1);
}

__m512 test_mm512_insertf32x4(__m512 __A, __m128 __B) {
  // CIR-LABEL: test_mm512_insertf32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_insertf32x4
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm512_insertf32x4
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_insertf32x4(__A, __B, 1);
}

__m512i test_mm512_inserti64x4(__m512i __A, __m256i __B) {
  // CIR-LABEL: test_mm512_inserti64x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s64i>

  // LLVM-LABEL: test_mm512_inserti64x4
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>

  // OGCG-LABEL: test_mm512_inserti64x4
  // OGCG: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_inserti64x4(__A, __B, 1);
}

__m512i test_mm512_inserti32x4(__m512i __A, __m128i __B) {
  // CIR-LABEL: test_mm512_inserti32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<16 x !s32i>

  // LLVM-LABEL: test_mm512_inserti32x4
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm512_inserti32x4
  // OGCG: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_inserti32x4(__A, __B, 1);
}

__mmask16 test_mm512_kand(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kand
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kand
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[RES:%.*]] = and <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> [[RES]] to i16

  // OGCG-LABEL: _mm512_kand
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: and <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kand(A, B);
}

__mmask16 test_mm512_kandn(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kandn
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kandn
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1> [[L]], splat (i1 true)
  // LLVM: and <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kandn
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: and <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kandn(A, B);
}

__mmask16 test_mm512_kor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: or <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: or <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kor(A, B);
}

__mmask16 test_mm512_kxnor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kxnor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kxnor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[NOT:%.*]] = xor <16 x i1> [[L]], splat (i1 true)
  // LLVM: [[RES:%.*]] = xor <16 x i1> [[NOT]], [[R]]
  // LLVM: bitcast <16 x i1> [[RES]] to i16

  // OGCG-LABEL: _mm512_kxnor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kxnor(A, B);
}

__mmask16 test_mm512_kxor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kxor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kxor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kxor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kxor(A, B);
}

__mmask16 test_mm512_knot(__mmask16 A) {
  // CIR-LABEL: _mm512_knot
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_knot
  // LLVM: bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_knot
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_knot(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask16 test_kmov_w(__mmask16 A) {
  // CIR-LABEL: test_kmov_w
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: test_kmov_w
  // LLVM: bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: test_kmov_w
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return __builtin_ia32_kmovw(A);
}

__mmask16 test_mm512_kunpackb(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kunpackb
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kunpackb
  // LLVM: [[A_VEC:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[B_VEC:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[A_HALF:%.*]] = shufflevector <16 x i1> [[A_VEC]], <16 x i1> [[A_VEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: [[B_HALF:%.*]] = shufflevector <16 x i1> [[B_VEC]], <16 x i1> [[B_VEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: [[RES:%.*]] = shufflevector <8 x i1> [[B_HALF]], <8 x i1> [[A_HALF]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: bitcast <16 x i1> [[RES]] to i16

  // OGCG-LABEL: _mm512_kunpackb
  // OGCG: [[A_VEC:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: [[B_VEC:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: [[A_HALF:%.*]] = shufflevector <16 x i1> [[A_VEC]], <16 x i1> [[A_VEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: [[B_HALF:%.*]] = shufflevector <16 x i1> [[B_VEC]], <16 x i1> [[B_VEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: [[RES:%.*]] = shufflevector <8 x i1> [[B_HALF]], <8 x i1> [[A_HALF]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: bitcast <16 x i1> [[RES]] to i16
  return _mm512_kunpackb(A, B);
}
__m256 test_mm512_i64gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_i64gather_ps
  // LLVM: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512

  // OGCG-LABEL: test_mm512_i64gather_ps
  // OGCG: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_i64gather_ps(__index, __addr, 2);
}

__m256 test_mm512_mask_i64gather_ps(__m256 __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_ps
  // LLVM: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512

  // OGCG-LABEL: test_mm512_mask_i64gather_ps
  // OGCG: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_mask_i64gather_ps(__v1_old, __mask, __index, __addr, 2);
}

__m256i test_mm512_i64gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_i64gather_epi32
  // LLVM: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512

  // OGCG-LABEL: test_mm512_i64gather_epi32
  // OGCG: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_i64gather_epi32(__index, __addr, 2);
}

__m256i test_mm512_mask_i64gather_epi32(__m256i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_epi32
  // LLVM: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512

  // OGCG-LABEL: test_mm512_mask_i64gather_epi32
  // OGCG: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_mask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2);
}

__m512d test_mm512_i64gather_pd(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpd.512

  // LLVM-LABEL: test_mm512_i64gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512

  // OGCG-LABEL: test_mm512_i64gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_i64gather_pd(__index, __addr, 2);
}

__m512d test_mm512_mask_i64gather_pd(__m512d __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpd.512

  // LLVM-LABEL: test_mm512_mask_i64gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512

  // OGCG-LABEL: test_mm512_mask_i64gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_mask_i64gather_pd(__v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i64gather_epi64(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpq.512

  // LLVM-LABEL: test_mm512_i64gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512

  // OGCG-LABEL: test_mm512_i64gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_i64gather_epi64(__index, __addr, 2);
}

__m512i test_mm512_mask_i64gather_epi64(__m512i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpq.512

  // LLVM-LABEL: test_mm512_mask_i64gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512

  // OGCG-LABEL: test_mm512_mask_i64gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_mask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2);
}

__m512 test_mm512_i32gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dps.512

  // LLVM-LABEL: test_mm512_i32gather_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512

  // OGCG-LABEL: test_mm512_i32gather_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_i32gather_ps(__index, __addr, 2);
}

__m512 test_mm512_mask_i32gather_ps(__m512 v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dps.512

  // LLVM-LABEL: test_mm512_mask_i32gather_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512

  // OGCG-LABEL: test_mm512_mask_i32gather_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_mask_i32gather_ps(v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i32gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpi.512

  // LLVM-LABEL: test_mm512_i32gather_epi32
  // LLVM: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512

  // OGCG-LABEL: test_mm512_i32gather_epi32
  // OGCG: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_i32gather_epi32(__index, __addr, 2);
}

__m512i test_mm512_mask_i32gather_epi32(__m512i __v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpi.512

  // LLVM-LABEL: test_mm512_mask_i32gather_epi32
  // LLVM: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512

  // OGCG-LABEL: test_mm512_mask_i32gather_epi32
  // OGCG: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_mask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2);
}

__m512d test_mm512_i32gather_pd(__m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpd.512

  // LLVM-LABEL: test_mm512_i32gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512

  // OGCG-LABEL: test_mm512_i32gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_i32gather_pd(__index, __addr, 2);
}

__m512d test_mm512_mask_i32gather_pd(__m512d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpd.512

  // LLVM-LABEL: test_mm512_mask_i32gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512

  // OGCG-LABEL: test_mm512_mask_i32gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_mask_i32gather_pd(__v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i32gather_epi64(__m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpq.512

  // LLVM-LABEL: test_mm512_i32gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
 
  // OGCG-LABEL: test_mm512_i32gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_i32gather_epi64(__index, __addr, 2);
}

__m512i test_mm512_mask_i32gather_epi64(__m512i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpq.512

  // LLVM-LABEL: test_mm512_mask_i32gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
 
  // OGCG-LABEL: test_mm512_mask_i32gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_mask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_ror_epi32(__m512i __A) {
  // CIR-LABEL: test_mm512_ror_epi32
  // CIR: cir.vec.splat %{{.*}} : !u32i, !cir.vector<16 x !u32i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i> 

  // LLVM-LABEL: test_mm512_ror_epi32
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %[[CASTED_VAR]], <16 x i32> %[[CASTED_VAR]], <16 x i32> splat (i32 5))

  // OGCG-LABEL: test_mm512_ror_epi32
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %[[CASTED_VAR]], <16 x i32> %[[CASTED_VAR]], <16 x i32> splat (i32 5))
  return _mm512_ror_epi32(__A, 5); 
}

__m512i test_mm512_ror_epi64(__m512i __A) {
  // CIR-LABEL: test_mm512_ror_epi64
  // CIR: cir.vec.splat %{{.*}} : !u64i, !cir.vector<8 x !u64i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i> 

  // LLVM-LABEL: test_mm512_ror_epi64
  // LLVM: %[[VAR:.*]] = load <8 x i64>, ptr %{{.*}}, align 64
  // LLVM: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %[[VAR]], <8 x i64> %[[VAR]], <8 x i64> splat (i64 5))

  // OGCG-LABEL: test_mm512_ror_epi64
  // OGCG: %[[VAR:.*]] = load <8 x i64>, ptr %{{.*}}, align 64
  // OGCG: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %[[VAR]], <8 x i64> %[[VAR]], <8 x i64> splat (i64 5))
  return _mm512_ror_epi64(__A, 5); 
}

void test_mm512_i32scatter_pd(void *__addr, __m256i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_i32scatter_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dpd.512"

  // LLVM-LABEL: test_mm512_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.dpd.512

  // OGCG-LABEL: test_mm512_i32scatter_pd
  // OGCG: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_i32scatter_pd(__addr, __index, __v1, 2);
}

void test_mm512_mask_i32scatter_pd(void *__addr, __mmask8 __mask, __m256i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dpd.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.dpd.512

  // OGCG-LABEL: test_mm512_mask_i32scatter_pd
  // OGCG: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_mask_i32scatter_pd(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i32scatter_ps(void *__addr, __m512i __index, __m512 __v1) {
  // CIR-LABEL: test_mm512_i32scatter_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dps.512"

  // LLVM-LABEL: test_mm512_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.dps.512

  // OGCG-LABEL: test_mm512_i32scatter_ps
  // OGCG: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_i32scatter_ps(__addr, __index, __v1, 2);
}

void test_mm512_mask_i32scatter_ps(void *__addr, __mmask16 __mask, __m512i __index, __m512 __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dps.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.dps.512

  // OGCG-LABEL: test_mm512_mask_i32scatter_ps
  // OGCG: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_mask_i32scatter_ps(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i64scatter_pd(void *__addr, __m512i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_i64scatter_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpd.512"

  // LLVM-LABEL: test_mm512_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.qpd.512

  // OGCG-LABEL: test_mm512_i64scatter_pd
  // OGCG: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_i64scatter_pd(__addr, __index, __v1, 2);
}

void test_mm512_mask_i64scatter_pd(void *__addr, __mmask8 __mask, __m512i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpd.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.qpd.512

  // OGCG-LABEL: test_mm512_mask_i64scatter_pd
  // OGCG: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_mask_i64scatter_pd(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i64scatter_ps(void *__addr, __m512i __index, __m256 __v1) {
  // CIR-LABEL: test_mm512_i64scatter_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qps.512"

  // LLVM-LABEL: test_mm512_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.qps.512

  // OGCG-LABEL: test_mm512_i64scatter_ps
  // OGCG: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_i64scatter_ps(__addr, __index, __v1, 2);
}

void test_mm512_mask_i64scatter_ps(void *__addr, __mmask8 __mask, __m512i __index, __m256 __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qps.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.qps.512

  // OGCG-LABEL: test_mm512_mask_i64scatter_ps
  // OGCG: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_mask_i64scatter_ps(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i32scatter_epi32(void *__addr, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_i32scatter_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dpi.512"

  // LLVM-LABEL: test_mm512_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.dpi.512

  // OGCG-LABEL: test_mm512_i32scatter_epi32
  // OGCG: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_i32scatter_epi32(__addr, __index, __v1, 2);
}

void test_mm512_mask_i32scatter_epi32(void *__addr, __mmask16 __mask, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.dpi.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.dpi.512

  // OGCG-LABEL: test_mm512_mask_i32scatter_epi32
  // OGCG: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_mask_i32scatter_epi32(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i64scatter_epi64(void *__addr, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_i64scatter_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpq.512"

  // LLVM-LABEL: test_mm512_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatter.qpq.512

  // OGCG-LABEL: test_mm512_i64scatter_epi64
  // OGCG: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_i64scatter_epi64(__addr, __index, __v1, 2);
}

void test_mm512_mask_i64scatter_epi64(void *__addr, __mmask8 __mask, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpq.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatter.qpq.512

  // OGCG-LABEL: test_mm512_mask_i64scatter_epi64
  // OGCG: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_mask_i64scatter_epi64(__addr, __mask, __index, __v1, 2);
}

void test_mm512_i64scatter_epi32(void *__addr, __m512i __index, __m256i __v1) {
  // CIR-LABEL: test_mm512_i64scatter_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpi.512"

  // LLVM-LABEL: test_mm512_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.qpi.512

  // OGCG-LABEL: test_mm512_i64scatter_epi32
  // OGCG: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_i64scatter_epi32(__addr, __index, __v1, 2);
}

void test_mm512_mask_i64scatter_epi32(void *__addr, __mmask8 __mask, __m512i __index, __m256i __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.scatter.qpi.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.qpi.512

  // OGCG-LABEL: test_mm512_mask_i64scatter_epi32
  // OGCG: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_mask_i64scatter_epi32(__addr, __mask, __index, __v1, 2);
}

__m256d test_mm512_extractf64x4_pd(__m512d a)
{
  // CIR-LABEL: test_mm512_extractf64x4_pd
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !cir.double>
  // CIR: cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !cir.double>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm512_extractf64x4_pd
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm512_extractf64x4_pd
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extractf64x4_pd(a, 1);
}

__m256d test_mm512_mask_extractf64x4_pd(__m256d  __W,__mmask8  __U,__m512d __A){
  // CIR-LABEL: test_mm512_mask_extractf64x4_pd
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !cir.double>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !cir.double>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm512_mask_extractf64x4_pd
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}

  // OGCG-LABEL: test_mm512_mask_extractf64x4_pd
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm512_mask_extractf64x4_pd( __W, __U, __A, 1);
}

__m256d test_mm512_maskz_extractf64x4_pd(__mmask8  __U,__m512d __A){
  // CIR-LABEL: test_mm512_maskz_extractf64x4_pd
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !cir.double>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !cir.double>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm512_maskz_extractf64x4_pd
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}

  // OGCG-LABEL: test_mm512_maskz_extractf64x4_pd
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm512_maskz_extractf64x4_pd( __U, __A, 1);
}

__m128 test_mm512_extractf32x4_ps(__m512 a)
{
  // CIR-LABEL: test_mm512_extractf32x4_ps
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !cir.float>
  // CIR: cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !cir.float>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm512_extractf32x4_ps
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm512_extractf32x4_ps
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extractf32x4_ps(a, 1);
}

__m128 test_mm512_mask_extractf32x4_ps(__m128 __W, __mmask8  __U,__m512 __A){
  // CIR-LABEL: test_mm512_mask_extractf32x4_ps
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !cir.float>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !cir.float>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.float>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm512_mask_extractf32x4_ps
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_mask_extractf32x4_ps
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm512_mask_extractf32x4_ps( __W, __U, __A, 1);
}

__m128 test_mm512_maskz_extractf32x4_ps( __mmask8  __U,__m512 __A){
  // CIR-LABEL: test_mm512_maskz_extractf32x4_ps
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !cir.float>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !cir.float>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.float>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm512_maskz_extractf32x4_ps
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_maskz_extractf32x4_ps
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm512_maskz_extractf32x4_ps(__U, __A, 1);
}

__m128i test_mm512_extracti32x4_epi32(__m512i __A) {
  // CIR-LABEL: test_mm512_extracti32x4_epi32
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !s32i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !s32i>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<4 x !s32i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s32i>

  // LLVM-LABEL: test_mm512_extracti32x4_epi32
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm512_extracti32x4_epi32
  // OGCG: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  return _mm512_extracti32x4_epi32(__A, 3);
}

__m128i test_mm512_mask_extracti32x4_epi32(__m128i __W, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: test_mm512_mask_extracti32x4_epi32
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !s32i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !s32i>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<4 x !s32i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s32i>

  // LLVM-LABEL: test_mm512_mask_extracti32x4_epi32
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // LLVM: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}

  // OGCG-LABEL: test_mm512_mask_extracti32x4_epi32
  // OGCG: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // OGCG: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm512_mask_extracti32x4_epi32(__W, __U, __A, 3);
}

__m128i test_mm512_maskz_extracti32x4_epi32(__mmask8 __U, __m512i __A) {
  // CIR-LABEL: test_mm512_maskz_extracti32x4_epi32
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<16 x !s32i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<16 x !s32i>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<4 x !s32i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s32i>

  // LLVM-LABEL: test_mm512_maskz_extracti32x4_epi32
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // LLVM: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}

  // OGCG-LABEL: test_mm512_maskz_extracti32x4_epi32
  // OGCG: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // OGCG: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm512_maskz_extracti32x4_epi32(__U, __A, 3);
}

__m256i test_mm512_extracti64x4_epi64(__m512i __A) {
  // CIR-LABEL: test_mm512_extracti64x4_epi64
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !s64i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !s64i>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s64i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_mm512_extracti64x4_epi64
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm512_extracti64x4_epi64
  // OGCG: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extracti64x4_epi64(__A, 1);
}

__m256i test_mm512_mask_extracti64x4_epi64(__m256i __W, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: test_mm512_mask_extracti64x4_epi64
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !s64i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !s64i>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s64i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_mm512_mask_extracti64x4_epi64
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}

  // OGCG-LABEL: test_mm512_mask_extracti64x4_epi64
  // OGCG: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm512_mask_extracti64x4_epi64(__W, __U, __A, 1);
}

__m256i test_mm512_maskz_extracti64x4_epi64(__mmask8 __U, __m512i __A) {
  // CIR-LABEL: test_mm512_maskz_extracti64x4_epi64
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<8 x !s64i>
  // CIR: [[FULL_VEC:%.*]] = cir.vec.shuffle(%{{.*}}, [[POISON]] : !cir.vector<8 x !s64i>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s64i>
  // CIR: [[MASK_VEC:%.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: [[FULL_MASK_VEC:%.*]] = cir.vec.shuffle([[MASK_VEC]], [[MASK_VEC]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary([[FULL_MASK_VEC]], [[FULL_VEC]], {{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_mm512_maskz_extracti64x4_epi64
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}

  // OGCG-LABEL: test_mm512_maskz_extracti64x4_epi64
  // OGCG: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // OGCG: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm512_maskz_extracti64x4_epi64(__U, __A, 1);
}

__m512i test_mm512_mul_epi32(__m512i __A, __m512i __B) {
  // CIR-LABEL: _mm512_mul_epi32
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<8 x !s64i>
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
  // CIR: [[BC_A:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !s64i>
  // CIR: [[BC_B:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !s64i>
  // CIR: [[MASK_SCALAR:%.*]] = cir.const #cir.int<4294967295> : !s64i
  // CIR: [[MASK_VEC:%.*]] = cir.vec.splat [[MASK_SCALAR]] : !s64i, !cir.vector<8 x !s64i>
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

int test_mm512_kortestc(__mmask16 __A, __mmask16 __B) {
  // CIR-LABEL: _mm512_kortestc
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<65535> : !u16i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]] = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ALL_ONES]]) : !u16i, !cir.bool
  // CIR: cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i

  // LLVM-LABEL: _mm512_kortestc
  // LLVM: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[OR:.*]] = or <16 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <16 x i1> %[[OR]] to i16
  // LLVM: %[[CMP:.*]] = icmp eq i16 %[[CAST]], -1
  // LLVM: zext i1 %[[CMP]] to i32

  // OGCG-LABEL: _mm512_kortestc
  // OGCG: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[OR:.*]] = or <16 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <16 x i1> %[[OR]] to i16
  // OGCG: %[[CMP:.*]] = icmp eq i16 %[[CAST]], -1
  // OGCG: zext i1 %[[CMP]] to i32
  return _mm512_kortestc(__A,__B);
}


int test_mm512_kortestz(__mmask16 __A, __mmask16 __B) {
  // CIR-LABEL: _mm512_kortestz
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u16i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]] = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ZERO]]) : !u16i, !cir.bool
  // CIR: cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i

  // LLVM-LABEL: _mm512_kortestz
  // LLVM: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[OR:.*]] = or <16 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <16 x i1> %[[OR]] to i16
  // LLVM: %[[CMP:.*]] = icmp eq i16 %[[CAST]], 0
  // LLVM: zext i1 %[[CMP]] to i32

  // OGCG-LABEL: _mm512_kortestz
  // OGCG: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[OR:.*]] = or <16 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <16 x i1> %[[OR]] to i16
  // OGCG: %[[CMP:.*]] = icmp eq i16 %[[CAST]], 0
  // OGCG: zext i1 %[[CMP]] to i32
  return _mm512_kortestz(__A,__B);
}

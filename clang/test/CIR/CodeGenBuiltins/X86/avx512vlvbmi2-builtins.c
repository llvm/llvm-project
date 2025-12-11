
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>


__m128i test_mm_mask_compress_epi16(__m128i __S, __mmask8 __U, __m128i __D) {
  // CIR-LABEL: test_mm_mask_compress_epi16
  // %[[MASK8:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.compress" %{{.+}}, %{{.+}}, %[[MASK8]]: (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<u, 1>>) -> !cir.vector<8 x !s16i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_mask_compress_epi16
  // %[[MASK8:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK8]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_mask_compress_epi16
  // %[[MASK8:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK8]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  return _mm_mask_compress_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi16(__mmask8 __U, __m128i __D) {
  // CIR-LABEL: test_mm_maskz_compress_epi16
  // %[[MASK8:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.compress" %{{.+}}, %{{.+}}, %[[MASK8]]: (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<u, 1>>) -> !cir.vector<8 x !s16i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_maskz_compress_epi16
  // %[[MASK8:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK8]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_maskz_compress_epi16
  // %[[MASK8:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK8]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  return _mm_maskz_compress_epi16(__U, __D);
}

__m128i test_mm_mask_compress_epi8(__m128i __S, __mmask16 __U, __m128i __D) {
  // CIR-LABEL: test_mm_mask_compress_epi8
  // %[[MASK16:.+]] = cir.cast bitcast %{{.+}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.compress" %{{.+}}, %{{.+}}, %[[MASK16]]: (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<u, 1>>) -> !cir.vector<16 x !s8i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_mask_compress_epi8
  // %[[MASK16:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK16]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_mask_compress_epi8
  // %[[MASK16:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK16]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  return _mm_mask_compress_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi8(__mmask16 __U, __m128i __D) {
  // CIR-LABEL: test_mm_maskz_compress_epi8
  // %[[ZERO:.+]] = cir.call @_mm_setzero_si128() : () -> !cir.vector<2 x !s64i>
  // %[[CAST1:.+]] = cir.cast bitcast %[[ZERO]] : !cir.vector<2 x !s64i> -> !cir.vector<16 x !s8i>
  // %[[MASK16:.+]] = cir.cast bitcast %{{.+}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.compress" %{{.+}}, %[[CAST1]], %[[MASK16]]: (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<u, 1>>) -> !cir.vector<16 x !s8i>
  // %[[CAST2:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_maskz_compress_epi8
  // store <2 x i64> zeroinitializer, ptr %{{.+}}, align 16
  // %[[CAST1:.+]] = bitcast <2 x i64> %{{.+}} to <16 x i8>
  // %[[MASK16:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.+}}, <16 x i8> %[[CAST1]], <16 x i1> %[[MASK16]])
  // %[[CAST2:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_maskz_compress_epi8
  // store <2 x i64> zeroinitializer, ptr %{{.+}}, align 16
  // %[[CAST1:.+]] = bitcast <2 x i64> %{{.+}} to <16 x i8>
  // %[[MASK16:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.+}}, <16 x i8> %[[CAST1]], <16 x i1> %[[MASK16]])
  // %[[CAST2:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  return _mm_maskz_compress_epi8(__U, __D);
}

__m128i test_mm_mask_expand_epi16(__m128i __S, __mmask8 __U, __m128i __D) {
  // CIR-LABEL: test_mm_mask_expand_epi16
  // %[[MASK16:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.expand" %{{.+}}, %{{.+}}, %[[MASK16]]: (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<u, 1>>) -> !cir.vector<8 x !s16i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_mask_expand_epi16
  // %[[MASK16:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK16]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_mask_expand_epi16
  // %[[MASK16:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK16]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  return _mm_mask_expand_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi16(__mmask8 __U, __m128i __D) {
  // CIR-LABEL: test_mm_maskz_expand_epi16
  // %[[MASK:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.expand" %{{.+}}, %{{.+}}, %[[MASK]]: (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !cir.int<u, 1>>) -> !cir.vector<8 x !s16i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_maskz_expand_epi16
  // %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_maskz_expand_epi16
  // %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // %[[RES:.+]] = call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <8 x i16> %[[RES]] to <2 x i64>

  return _mm_maskz_expand_epi16(__U, __D);
}

__m128i test_mm_mask_expand_epi8(__m128i __S, __mmask16 __U, __m128i __D) {
  // CIR-LABEL: test_mm_mask_expand_epi8
  // %[[MASK:.+]] = cir.cast bitcast %{{.+}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.expand" %{{.+}}, %{{.+}}, %[[MASK]]: (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<u, 1>>) -> !cir.vector<16 x !s8i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_mask_expand_epi8
  // %[[MASK:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_mask_expand_epi8
  // %[[MASK:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  return _mm_mask_expand_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi8(__mmask16 __U, __m128i __D) {
  // CIR-LABEL: test_mm_maskz_expand_epi8
  // %[[MASK:.+]] = cir.cast bitcast %{{.+}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // %[[RES:.+]] = cir.call_llvm_intrinsic "x86.avx512.mask.expand" %{{.+}}, %{{.+}}, %[[MASK]]: (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !cir.int<u, 1>>) -> !cir.vector<16 x !s8i>
  // %[[CAST:.+]] = cir.cast bitcast %[[RES]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_maskz_expand_epi8
  // %[[MASK:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_maskz_expand_epi8
  // %[[MASK:.+]] = bitcast i16 %{{.+}} to <16 x i1>
  // %[[RES:.+]] = call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i1> %[[MASK]])
  // %[[CAST:.+]] = bitcast <16 x i8> %[[RES]] to <2 x i64>

  return _mm_maskz_expand_epi8(__U, __D);
}

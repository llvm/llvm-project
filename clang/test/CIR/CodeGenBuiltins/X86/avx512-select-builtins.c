// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// REQUIRES: avx512bw
// REQUIRES: avx512vl

#include <immintrin.h>

// CIR-LABEL: test_selectb_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i8>, <16 x i8>* %
// LLVM: %[[B:.*]] = load <16 x i8>, <16 x i8>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i8> %[[A]], <16 x i8> %[[B]]
// LLVM: store <16 x i8> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i8>, <16 x i8>* %
// OGCG: %[[B:.*]] = load <16 x i8>, <16 x i8>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i8> %[[A]], <16 x i8> %[[B]]
// OGCG: store <16 x i8> %[[RES]]
__m128i test_selectb_128(__mmask16 k, __m128i a, __m128i b) {
  return _mm_selectb_128(k, a, b);
}

// CIR-LABEL: test_selectb_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// LLVM: %[[A:.*]] = load <32 x i8>, <32 x i8>* %
// LLVM: %[[B:.*]] = load <32 x i8>, <32 x i8>* %
// LLVM: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i8> %[[A]], <32 x i8> %[[B]]
// LLVM: store <32 x i8> %[[RES]]
// OGCG: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// OGCG: %[[A:.*]] = load <32 x i8>, <32 x i8>* %
// OGCG: %[[B:.*]] = load <32 x i8>, <32 x i8>* %
// OGCG: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i8> %[[A]], <32 x i8> %[[B]]
// OGCG: store <32 x i8> %[[RES]]
__m256i test_selectb_256(__mmask32 k, __m256i a, __m256i b) {
  return _mm256_selectb_epi8(k, a, b);
}

// CIR-LABEL: test_selectb_512
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <64 x i1>, <64 x i1>* %
// LLVM: %[[A:.*]] = load <64 x i8>, <64 x i8>* %
// LLVM: %[[B:.*]] = load <64 x i8>, <64 x i8>* %
// LLVM: %[[RES:.*]] = select <64 x i1> %[[MASK]], <64 x i8> %[[A]], <64 x i8> %[[B]]
// LLVM: store <64 x i8> %[[RES]]
// OGCG: %[[MASK:.*]] = load <64 x i1>, <64 x i1>* %
// OGCG: %[[A:.*]] = load <64 x i8>, <64 x i8>* %
// OGCG: %[[B:.*]] = load <64 x i8>, <64 x i8>* %
// OGCG: %[[RES:.*]] = select <64 x i1> %[[MASK]], <64 x i8> %[[A]], <64 x i8> %[[B]]
// OGCG: store <64 x i8> %[[RES]]
__m512i test_selectb_512(__mmask64 k, __m512i a, __m512i b) {
  return _mm512_selectb_epi8(k, a, b);
}

// CIR-LABEL: test_selectw_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// LLVM: store <8 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// OGCG: store <8 x i16> %[[RES]]
__m128i test_selectw_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectw_128(k, a, b);
}

// CIR-LABEL: test_selectw_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// LLVM: store <16 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// OGCG: store <16 x i16> %[[RES]]
__m256i test_selectw_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectw_epi16(k, a, b);
}

// CIR-LABEL: test_selectw_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectw_512
// LLVM: select <32 x i16>
// OGCG-LABEL: test_selectw_512
// OGCG: select <32 x i16>
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i32>, <16 x i32>* %
// LLVM: %[[B:.*]] = load <16 x i32>, <16 x i32>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i32> %[[A]], <16 x i32> %[[B]]
// LLVM: store <16 x i32> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i32>, <16 x i32>* %
// OGCG: %[[B:.*]] = load <16 x i32>, <16 x i32>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i32> %[[A]], <16 x i32> %[[B]]
// OGCG: store <16 x i32> %[[RES]]
__m512i test_selectw_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectw_epi16(k, a, b);
}

// CIR-LABEL: test_selectd_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// LLVM: %[[A:.*]] = load <4 x i32>, <4 x i32>* %
// LLVM: %[[B:.*]] = load <4 x i32>, <4 x i32>* %
// LLVM: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x i32> %[[A]], <4 x i32> %[[B]]
// LLVM: store <4 x i32> %[[RES]]
// OGCG: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// OGCG: %[[A:.*]] = load <4 x i32>, <4 x i32>* %
// OGCG: %[[B:.*]] = load <4 x i32>, <4 x i32>* %
// OGCG: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x i32> %[[A]], <4 x i32> %[[B]]
// OGCG: store <4 x i32> %[[RES]]
__m128i test_selectd_128(__mmask4 k, __m128i a, __m128i b) {
  return _mm_selectd_128(k, a, b);
}

// CIR-LABEL: test_selectd_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x i32>, <8 x i32>* %
// LLVM: %[[B:.*]] = load <8 x i32>, <8 x i32>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i32> %[[A]], <8 x i32> %[[B]]
// LLVM: store <8 x i32> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x i32>, <8 x i32>* %
// OGCG: %[[B:.*]] = load <8 x i32>, <8 x i32>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i32> %[[A]], <8 x i32> %[[B]]
// OGCG: store <8 x i32> %[[RES]]
__m256i test_selectd_256(__mmask8 k, __m256i a, __m256i b) {
  return _mm256_selectd_epi32(k, a, b);
}

// CIR-LABEL: test_selectd_512
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i32>, <16 x i32>* %
// LLVM: %[[B:.*]] = load <16 x i32>, <16 x i32>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i32> %[[A]], <16 x i32> %[[B]]
// LLVM: store <16 x i32> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i32>, <16 x i32>* %
// OGCG: %[[B:.*]] = load <16 x i32>, <16 x i32>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i32> %[[A]], <16 x i32> %[[B]]
// OGCG: store <16 x i32> %[[RES]]
__m512i test_selectd_512(__mmask16 k, __m512i a, __m512i b) {
  return _mm512_selectd_epi32(k, a, b);
}

// CIR-LABEL: test_selectq_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <2 x i1>, <2 x i1>* %
// LLVM: %[[A:.*]] = load <2 x i64>, <2 x i64>* %
// LLVM: %[[B:.*]] = load <2 x i64>, <2 x i64>* %
// LLVM: %[[RES:.*]] = select <2 x i1> %[[MASK]], <2 x i64> %[[A]], <2 x i64> %[[B]]
// LLVM: store <2 x i64> %[[RES]]
// OGCG: %[[MASK:.*]] = load <2 x i1>, <2 x i1>* %
// OGCG: %[[A:.*]] = load <2 x i64>, <2 x i64>* %
// OGCG: %[[B:.*]] = load <2 x i64>, <2 x i64>* %
// OGCG: %[[RES:.*]] = select <2 x i1> %[[MASK]], <2 x i64> %[[A]], <2 x i64> %[[B]]
// OGCG: store <2 x i64> %[[RES]]
__m128i test_selectq_128(__mmask2 k, __m128i a, __m128i b) {
  return _mm_selectq_128(k, a, b);
}

// CIR-LABEL: test_selectq_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// LLVM: %[[A:.*]] = load <4 x i64>, <4 x i64>* %
// LLVM: %[[B:.*]] = load <4 x i64>, <4 x i64>* %
// LLVM: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x i64> %[[A]], <4 x i64> %[[B]]
// LLVM: store <4 x i64> %[[RES]]
// OGCG: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// OGCG: %[[A:.*]] = load <4 x i64>, <4 x i64>* %
// OGCG: %[[B:.*]] = load <4 x i64>, <4 x i64>* %
// OGCG: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x i64> %[[A]], <4 x i64> %[[B]]
// OGCG: store <4 x i64> %[[RES]]
__m256i test_selectq_256(__mmask4 k, __m256i a, __m256i b) {
  return _mm256_selectq_epi64(k, a, b);
}

// CIR-LABEL: test_selectq_512
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x i64>, <8 x i64>* %
// LLVM: %[[B:.*]] = load <8 x i64>, <8 x i64>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i64> %[[A]], <8 x i64> %[[B]]
// LLVM: store <8 x i64> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x i64>, <8 x i64>* %
// OGCG: %[[B:.*]] = load <8 x i64>, <8 x i64>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i64> %[[A]], <8 x i64> %[[B]]
// OGCG: store <8 x i64> %[[RES]]
__m512i test_selectq_512(__mmask8 k, __m512i a, __m512i b) {
  return _mm512_selectq_epi64(k, a, b);
}

// CIR-LABEL: test_selectph_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// LLVM: store <8 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// OGCG: store <8 x i16> %[[RES]]
__m128i test_selectph_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectph_128(k, a, b);
}

// CIR-LABEL: test_selectph_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// LLVM: store <16 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// OGCG: store <16 x i16> %[[RES]]
__m256i test_selectph_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectph_epi16(k, a, b);
}

// CIR-LABEL: test_selectph_512
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// LLVM: %[[A:.*]] = load <32 x i16>, <32 x i16>* %
// LLVM: %[[B:.*]] = load <32 x i16>, <32 x i16>* %
// LLVM: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i16> %[[A]], <32 x i16> %[[B]]
// LLVM: store <32 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// OGCG: %[[A:.*]] = load <32 x i16>, <32 x i16>* %
// OGCG: %[[B:.*]] = load <32 x i16>, <32 x i16>* %
// OGCG: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i16> %[[A]], <32 x i16> %[[B]]
// OGCG: store <32 x i16> %[[RES]]
__m512i test_selectph_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectph_epi16(k, a, b);
}

// CIR-LABEL: test_selectpbf_128
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// LLVM: store <8 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// OGCG: store <8 x i16> %[[RES]]
__m128i test_selectpbf_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectpbf_128(k, a, b);
}

// CIR-LABEL: test_selectpbf_256
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// LLVM: store <16 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[B:.*]] = load <16 x i16>, <16 x i16>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x i16> %[[A]], <16 x i16> %[[B]]
// OGCG: store <16 x i16> %[[RES]]
__m256i test_selectpbf_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectpbf_epi16(k, a, b);
}

// CIR-LABEL: test_selectpbf_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpbf_512
// LLVM: select
// OGCG-LABEL: test_selectpbf_512
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// LLVM: %[[A:.*]] = load <32 x i16>, <32 x i16>* %
// LLVM: %[[B:.*]] = load <32 x i16>, <32 x i16>* %
// LLVM: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i16> %[[A]], <32 x i16> %[[B]]
// LLVM: store <32 x i16> %[[RES]]
// OGCG: %[[MASK:.*]] = load <32 x i1>, <32 x i1>* %
// OGCG: %[[A:.*]] = load <32 x i16>, <32 x i16>* %
// OGCG: %[[B:.*]] = load <32 x i16>, <32 x i16>* %
// OGCG: %[[RES:.*]] = select <32 x i1> %[[MASK]], <32 x i16> %[[A]], <32 x i16> %[[B]]
// OGCG: store <32 x i16> %[[RES]]
__m512i test_selectpbf_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectpbf_epi16(k, a, b);
}

// CIR-LABEL: test_selectps_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_128
// LLVM: select
// OGCG-LABEL: test_selectps_128
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// LLVM: %[[A:.*]] = load <4 x float>, <4 x float>* %
// LLVM: %[[B:.*]] = load <4 x float>, <4 x float>* %
// LLVM: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[A]], <4 x float> %[[B]]
// LLVM: store <4 x float> %[[RES]]
// OGCG: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// OGCG: %[[A:.*]] = load <4 x float>, <4 x float>* %
// OGCG: %[[B:.*]] = load <4 x float>, <4 x float>* %
// OGCG: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[A]], <4 x float> %[[B]]
// OGCG: store <4 x float> %[[RES]]
__m128 test_selectps_128(__mmask8 k, __m128 a, __m128 b) {
  return _mm_selectps_128(k, a, b);
}

// CIR-LABEL: test_selectps_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_256
// LLVM: select
// OGCG-LABEL: test_selectps_256
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x float>, <8 x float>* %
// LLVM: %[[B:.*]] = load <8 x float>, <8 x float>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[A]], <8 x float> %[[B]]
// LLVM: store <8 x float> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x float>, <8 x float>* %
// OGCG: %[[B:.*]] = load <8 x float>, <8 x float>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[A]], <8 x float> %[[B]]
// OGCG: store <8 x float> %[[RES]]
__m256 test_selectps_256(__mmask8 k, __m256 a, __m256 b) {
  return _mm256_selectps(k, a, b);
}

// CIR-LABEL: test_selectps_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_512
// LLVM: select
// OGCG-LABEL: test_selectps_512
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// LLVM: %[[A:.*]] = load <16 x float>, <16 x float>* %
// LLVM: %[[B:.*]] = load <16 x float>, <16 x float>* %
// LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[A]], <16 x float> %[[B]]
// LLVM: store <16 x float> %[[RES]]
// OGCG: %[[MASK:.*]] = load <16 x i1>, <16 x i1>* %
// OGCG: %[[A:.*]] = load <16 x float>, <16 x float>* %
// OGCG: %[[B:.*]] = load <16 x float>, <16 x float>* %
// OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[A]], <16 x float> %[[B]]
// OGCG: store <16 x float> %[[RES]]
__m512 test_selectps_512(__mmask16 k, __m512 a, __m512 b) {
  return _mm512_selectps(k, a, b);
}

// CIR-LABEL: test_selectpd_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_128
// LLVM: select
// OGCG-LABEL: test_selectpd_128
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <2 x i1>, <2 x i1>* %
// LLVM: %[[A:.*]] = load <2 x double>, <2 x double>* %
// LLVM: %[[B:.*]] = load <2 x double>, <2 x double>* %
// LLVM: %[[RES:.*]] = select <2 x i1> %[[MASK]], <2 x double> %[[A]], <2 x double> %[[B]]
// LLVM: store <2 x double> %[[RES]]
// OGCG: %[[MASK:.*]] = load <2 x i1>, <2 x i1>* %
// OGCG: %[[A:.*]] = load <2 x double>, <2 x double>* %
// OGCG: %[[B:.*]] = load <2 x double>, <2 x double>* %
// OGCG: %[[RES:.*]] = select <2 x i1> %[[MASK]], <2 x double> %[[A]], <2 x double> %[[B]]
// OGCG: store <2 x double> %[[RES]]
__m128d test_selectpd_128(__mmask8 k, __m128d a, __m128d b) {
  return _mm_selectpd_128(k, a, b);
}

// CIR-LABEL: test_selectpd_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_256
// LLVM: select
// OGCG-LABEL: test_selectpd_256
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// LLVM: %[[A:.*]] = load <4 x double>, <4 x double>* %
// LLVM: %[[B:.*]] = load <4 x double>, <4 x double>* %
// LLVM: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x double> %[[A]], <4 x double> %[[B]]
// LLVM: store <4 x double> %[[RES]]
// OGCG: %[[MASK:.*]] = load <4 x i1>, <4 x i1>* %
// OGCG: %[[A:.*]] = load <4 x double>, <4 x double>* %
// OGCG: %[[B:.*]] = load <4 x double>, <4 x double>* %
// OGCG: %[[RES:.*]] = select <4 x i1> %[[MASK]], <4 x double> %[[A]], <4 x double> %[[B]]
// OGCG: store <4 x double> %[[RES]]
__m256d test_selectpd_256(__mmask8 k, __m256d a, __m256d b) {
  return _mm256_selectpd(k, a, b);
}

// CIR-LABEL: test_selectpd_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_512
// LLVM: select
// OGCG-LABEL: test_selectpd_512
// OGCG: select
// CIR: %[[MASK:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[RES:.*]] = cir.vec.ternary %[[MASK]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// LLVM: %[[A:.*]] = load <8 x double>, <8 x double>* %
// LLVM: %[[B:.*]] = load <8 x double>, <8 x double>* %
// LLVM: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x double> %[[A]], <8 x double> %[[B]]
// LLVM: store <8 x double> %[[RES]]
// OGCG: %[[MASK:.*]] = load <8 x i1>, <8 x i1>* %
// OGCG: %[[A:.*]] = load <8 x double>, <8 x double>* %
// OGCG: %[[B:.*]] = load <8 x double>, <8 x double>* %
// OGCG: %[[RES:.*]] = select <8 x i1> %[[MASK]], <8 x double> %[[A]], <8 x double> %[[B]]
// OGCG: store <8 x double> %[[RES]]
__m512d test_selectpd_512(__mmask8 k, __m512d a, __m512d b) {
  return _mm512_selectpd(k, a, b);
}

// CIR-LABEL: test_selectsh_128
// CIR: cir.cmp {{.*}} ne
// CIR: cir.select
// LLVM-LABEL: test_selectsh_128
// LLVM: select
// OGCG-LABEL: test_selectsh_128
// OGCG: select
// CIR: %[[K:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[CMP:.*]] = cir.cmp ne, %[[K]], 0
// CIR: %[[RES:.*]] = cir.select %[[CMP]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[K:.*]] = load i16, i16* %
// LLVM: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// LLVM: %[[RES:.*]] = select i1 %[[CMP]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// LLVM: store <8 x i16> %[[RES]]
// OGCG: %[[K:.*]] = load i16, i16* %
// OGCG: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// OGCG: %[[RES:.*]] = select i1 %[[CMP]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// OGCG: store <8 x i16> %[[RES]]
__m128i test_selectsh_128(unsigned short k, __m128i a, __m128i b) {
  return _mm_selectsh_128(k, a, b);
}

// CIR-LABEL: test_selectsbf_128
// CIR: cir.cmp {{.*}} ne
// CIR: cir.select
// LLVM-LABEL: test_selectsbf_128
// LLVM: select
// OGCG-LABEL: test_selectsbf_128
// OGCG: select
// CIR: %[[K:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[CMP:.*]] = cir.cmp ne, %[[K]], 0
// CIR: %[[RES:.*]] = cir.select %[[CMP]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[K:.*]] = load i16, i16* %
// LLVM: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// LLVM: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// LLVM: %[[RES:.*]] = select i1 %[[CMP]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// LLVM: store <8 x i16> %[[RES]]
// OGCG: %[[K:.*]] = load i16, i16* %
// OGCG: %[[A:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[B:.*]] = load <8 x i16>, <8 x i16>* %
// OGCG: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// OGCG: %[[RES:.*]] = select i1 %[[CMP]], <8 x i16> %[[A]], <8 x i16> %[[B]]
// OGCG: store <8 x i16> %[[RES]]
__m128i test_selectsbf_128(unsigned short k, __m128i a, __m128i b) {
  return _mm_selectsbf_128(k, a, b);
}

// CIR-LABEL: test_selectss_128
// CIR: cir.cmp {{.*}} ne
// CIR: cir.select
// LLVM-LABEL: test_selectss_128
// LLVM: select
// OGCG-LABEL: test_selectss_128
// OGCG: select
// CIR: %[[K:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[CMP:.*]] = cir.cmp ne, %[[K]], 0
// CIR: %[[RES:.*]] = cir.select %[[CMP]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[K:.*]] = load i16, i16* %
// LLVM: %[[A:.*]] = load <4 x float>, <4 x float>* %
// LLVM: %[[B:.*]] = load <4 x float>, <4 x float>* %
// LLVM: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// LLVM: %[[RES:.*]] = select i1 %[[CMP]], <4 x float> %[[A]], <4 x float> %[[B]]
// LLVM: store <4 x float> %[[RES]]
// OGCG: %[[K:.*]] = load i16, i16* %
// OGCG: %[[A:.*]] = load <4 x float>, <4 x float>* %
// OGCG: %[[B:.*]] = load <4 x float>, <4 x float>* %
// OGCG: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// OGCG: %[[RES:.*]] = select i1 %[[CMP]], <4 x float> %[[A]], <4 x float> %[[B]]
// OGCG: store <4 x float> %[[RES]]
__m128 test_selectss_128(unsigned short k, __m128 a, __m128 b) {
  return _mm_selectss_128(k, a, b);
}

// CIR-LABEL: test_selectsd_128
// CIR: cir.cmp {{.*}} ne
// CIR: cir.select
// LLVM-LABEL: test_selectsd_128
// LLVM: select
// OGCG-LABEL: test_selectsd_128
// OGCG: select
// CIR: %[[K:.*]] = cir.load{{.*}}%arg0
// CIR: %[[A:.*]] = cir.load{{.*}}%arg1
// CIR: %[[B:.*]] = cir.load{{.*}}%arg2
// CIR: %[[CMP:.*]] = cir.cmp ne, %[[K]], 0
// CIR: %[[RES:.*]] = cir.select %[[CMP]], %[[A]], %[[B]]
// CIR: cir.store %[[RES]]
// LLVM: %[[K:.*]] = load i16, i16* %
// LLVM: %[[A:.*]] = load <2 x double>, <2 x double>* %
// LLVM: %[[B:.*]] = load <2 x double>, <2 x double>* %
// LLVM: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// LLVM: %[[RES:.*]] = select i1 %[[CMP]], <2 x double> %[[A]], <2 x double> %[[B]]
// LLVM: store <2 x double> %[[RES]]
// OGCG: %[[K:.*]] = load i16, i16* %
// OGCG: %[[A:.*]] = load <2 x double>, <2 x double>* %
// OGCG: %[[B:.*]] = load <2 x double>, <2 x double>* %
// OGCG: %[[CMP:.*]] = icmp ne i16 %[[K]], 0
// OGCG: %[[RES:.*]] = select i1 %[[CMP]], <2 x double> %[[A]], <2 x double> %[[B]]
// OGCG: store <2 x double> %[[RES]]
__m128d test_selectsd_128(unsigned short k, __m128d a, __m128d b) {
  return _mm_selectsd_128(k, a, b);
}

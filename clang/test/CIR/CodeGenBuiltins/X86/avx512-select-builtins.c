// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512bw -target-feature +avx512vl -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// REQUIRES: avx512bw
// REQUIRES: avx512vl

#include <immintrin.h>

// CIR-LABEL: test_selectb_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectb_128
// LLVM: select <16 x i8>
// OGCG-LABEL: test_selectb_128
// OGCG: select <16 x i8>
__m128i test_selectb_128(__mmask16 k, __m128i a, __m128i b) {
  return _mm_selectb_128(k, a, b);
}

// CIR-LABEL: test_selectb_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectb_256
// LLVM: select <32 x i8>
// OGCG-LABEL: test_selectb_256
// OGCG: select <32 x i8>
__m256i test_selectb_256(__mmask32 k, __m256i a, __m256i b) {
  return _mm256_selectb_epi8(k, a, b);
}

// CIR-LABEL: test_selectb_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectb_512
// LLVM: select <64 x i8>
// OGCG-LABEL: test_selectb_512
// OGCG: select <64 x i8>
__m512i test_selectb_512(__mmask64 k, __m512i a, __m512i b) {
  return _mm512_selectb_epi8(k, a, b);
}

// CIR-LABEL: test_selectw_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectw_128
// LLVM: select <8 x i16>
// OGCG-LABEL: test_selectw_128
// OGCG: select <8 x i16>
__m128i test_selectw_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectw_128(k, a, b);
}

// CIR-LABEL: test_selectw_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectw_256
// LLVM: select <16 x i16>
// OGCG-LABEL: test_selectw_256
// OGCG: select <16 x i16>
__m256i test_selectw_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectw_epi16(k, a, b);
}

// CIR-LABEL: test_selectw_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectw_512
// LLVM: select <32 x i16>
// OGCG-LABEL: test_selectw_512
// OGCG: select <32 x i16>
__m512i test_selectw_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectw_epi16(k, a, b);
}

// CIR-LABEL: test_selectd_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectd_128
// LLVM: select <4 x i32>
// OGCG-LABEL: test_selectd_128
// OGCG: select <4 x i32>
__m128i test_selectd_128(__mmask4 k, __m128i a, __m128i b) {
  return _mm_selectd_128(k, a, b);
}

// CIR-LABEL: test_selectd_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectd_256
// LLVM: select <8 x i32>
// OGCG-LABEL: test_selectd_256
// OGCG: select <8 x i32>
__m256i test_selectd_256(__mmask8 k, __m256i a, __m256i b) {
  return _mm256_selectd_epi32(k, a, b);
}

// CIR-LABEL: test_selectd_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectd_512
// LLVM: select <16 x i32>
// OGCG-LABEL: test_selectd_512
// OGCG: select <16 x i32>
__m512i test_selectd_512(__mmask16 k, __m512i a, __m512i b) {
  return _mm512_selectd_epi32(k, a, b);
}

// CIR-LABEL: test_selectq_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectq_128
// LLVM: select <2 x i64>
// OGCG-LABEL: test_selectq_128
// OGCG: select <2 x i64>
__m128i test_selectq_128(__mmask2 k, __m128i a, __m128i b) {
  return _mm_selectq_128(k, a, b);
}

// CIR-LABEL: test_selectq_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectq_256
// LLVM: select <4 x i64>
// OGCG-LABEL: test_selectq_256
// OGCG: select <4 x i64>
__m256i test_selectq_256(__mmask4 k, __m256i a, __m256i b) {
  return _mm256_selectq_epi64(k, a, b);
}

// CIR-LABEL: test_selectq_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectq_512
// LLVM: select <8 x i64>
// OGCG-LABEL: test_selectq_512
// OGCG: select <8 x i64>
__m512i test_selectq_512(__mmask8 k, __m512i a, __m512i b) {
  return _mm512_selectq_epi64(k, a, b);
}

// CIR-LABEL: test_selectph_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectph_128
// LLVM: select
// OGCG-LABEL: test_selectph_128
// OGCG: select
__m128i test_selectph_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectph_128(k, a, b);
}

// CIR-LABEL: test_selectph_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectph_256
// LLVM: select
// OGCG-LABEL: test_selectph_256
// OGCG: select
__m256i test_selectph_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectph_epi16(k, a, b);
}

// CIR-LABEL: test_selectph_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectph_512
// LLVM: select
// OGCG-LABEL: test_selectph_512
// OGCG: select
__m512i test_selectph_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectph_epi16(k, a, b);
}

// CIR-LABEL: test_selectpbf_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpbf_128
// LLVM: select
// OGCG-LABEL: test_selectpbf_128
// OGCG: select
__m128i test_selectpbf_128(__mmask8 k, __m128i a, __m128i b) {
  return _mm_selectpbf_128(k, a, b);
}

// CIR-LABEL: test_selectpbf_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpbf_256
// LLVM: select
// OGCG-LABEL: test_selectpbf_256
// OGCG: select
__m256i test_selectpbf_256(__mmask16 k, __m256i a, __m256i b) {
  return _mm256_selectpbf_epi16(k, a, b);
}

// CIR-LABEL: test_selectpbf_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpbf_512
// LLVM: select
// OGCG-LABEL: test_selectpbf_512
// OGCG: select
__m512i test_selectpbf_512(__mmask32 k, __m512i a, __m512i b) {
  return _mm512_selectpbf_epi16(k, a, b);
}

// CIR-LABEL: test_selectps_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_128
// LLVM: select
// OGCG-LABEL: test_selectps_128
// OGCG: select
__m128 test_selectps_128(__mmask8 k, __m128 a, __m128 b) {
  return _mm_selectps_128(k, a, b);
}

// CIR-LABEL: test_selectps_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_256
// LLVM: select
// OGCG-LABEL: test_selectps_256
// OGCG: select
__m256 test_selectps_256(__mmask8 k, __m256 a, __m256 b) {
  return _mm256_selectps(k, a, b);
}

// CIR-LABEL: test_selectps_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectps_512
// LLVM: select
// OGCG-LABEL: test_selectps_512
// OGCG: select
__m512 test_selectps_512(__mmask16 k, __m512 a, __m512 b) {
  return _mm512_selectps(k, a, b);
}

// CIR-LABEL: test_selectpd_128
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_128
// LLVM: select
// OGCG-LABEL: test_selectpd_128
// OGCG: select
__m128d test_selectpd_128(__mmask8 k, __m128d a, __m128d b) {
  return _mm_selectpd_128(k, a, b);
}

// CIR-LABEL: test_selectpd_256
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_256
// LLVM: select
// OGCG-LABEL: test_selectpd_256
// OGCG: select
__m256d test_selectpd_256(__mmask8 k, __m256d a, __m256d b) {
  return _mm256_selectpd(k, a, b);
}

// CIR-LABEL: test_selectpd_512
// CIR: cir.vec.ternary
// LLVM-LABEL: test_selectpd_512
// LLVM: select
// OGCG-LABEL: test_selectpd_512
// OGCG: select
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
__m128d test_selectsd_128(unsigned short k, __m128d a, __m128d b) {
  return _mm_selectsd_128(k, a, b);
}

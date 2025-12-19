// RUN: %clang_cc1 -fclangir -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

#include <immintrin.h>

// CIR-LABEL: test_mm512_mask_cvtneps_pbh
// CIR: cir.call @_mm512_mask_cvtneps_pbh({{.*}}, {{.*}}, {{.*}})
// LLVM-LABEL: test_mm512_mask_cvtneps_pbh
// LLVM: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
// OGCG-LABEL: test_mm512_mask_cvtneps_pbh
// OGCG: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
__m256bh test_mm512_mask_cvtneps_pbh(__m256bh src, __mmask16 k, __m512 a) {
  return _mm512_mask_cvtneps_pbh(src, k, a);
}

// CIR-LABEL: test_mm512_maskz_cvtneps_pbh
// CIR: cir.call @_mm512_maskz_cvtneps_pbh({{.*}}, {{.*}})
// LLVM-LABEL: test_mm512_maskz_cvtneps_pbh
// LLVM: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
// OGCG-LABEL: test_mm512_maskz_cvtneps_pbh
// OGCG: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
__m256bh test_mm512_maskz_cvtneps_pbh(__mmask16 k, __m512 a) {
  return _mm512_maskz_cvtneps_pbh(k, a);
}

// CIR-LABEL: test_mm256_mask_cvtneps_pbh
// CIR: cir.call @_mm256_mask_cvtneps_pbh({{.*}}, {{.*}}, {{.*}})
// LLVM-LABEL: test_mm256_mask_cvtneps_pbh
// LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
// OGCG-LABEL: test_mm256_mask_cvtneps_pbh
// OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
__m128bh test_mm256_mask_cvtneps_pbh(__m128bh src, __mmask8 k, __m256 a) {
  return _mm256_mask_cvtneps_pbh(src, k, a);
}

// CIR-LABEL: test_mm256_maskz_cvtneps_pbh
// CIR: cir.call @_mm256_maskz_cvtneps_pbh({{.*}}, {{.*}})
// LLVM-LABEL: test_mm256_maskz_cvtneps_pbh
// LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
// OGCG-LABEL: test_mm256_maskz_cvtneps_pbh
// OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
__m128bh test_mm256_maskz_cvtneps_pbh(__mmask8 k, __m256 a) {
  return _mm256_maskz_cvtneps_pbh(k, a);
}

// CIR-LABEL: test_mm_mask_cvtneps_pbh
// CIR: cir.call @_mm_mask_cvtneps_pbh({{.*}}, {{.*}}, {{.*}})
// LLVM-LABEL: test_mm_mask_cvtneps_pbh
// LLVM: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
// OGCG-LABEL: test_mm_mask_cvtneps_pbh
// OGCG: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
__m64bh test_mm_mask_cvtneps_pbh(__m64bh src, __mmask8 k, __m128 a) {
  return _mm_mask_cvtneps_pbh(src, k, a);
}

// CIR-LABEL: test_mm_maskz_cvtneps_pbh
// CIR: cir.call @_mm_maskz_cvtneps_pbh({{.*}}, {{.*}})
// LLVM-LABEL: test_mm_maskz_cvtneps_pbh
// LLVM: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
// OGCG-LABEL: test_mm_maskz_cvtneps_pbh
// OGCG: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
__m64bh test_mm_maskz_cvtneps_pbh(__mmask8 k, __m128 a) {
  return _mm_maskz_cvtneps_pbh(k, a);
}

// CIR-LABEL: test_mm512_cvtneps_pbh
// CIR: cir.call @_mm512_cvtneps_pbh({{.*}})
// LLVM-LABEL: test_mm512_cvtneps_pbh
// LLVM: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
// OGCG-LABEL: test_mm512_cvtneps_pbh
// OGCG: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
__m256bh test_mm512_cvtneps_pbh(__m512 a) {
  return _mm512_cvtneps_pbh(a);
}

// CIR-LABEL: test_mm256_cvtneps_pbh
// CIR: cir.call @_mm256_cvtneps_pbh({{.*}})
// LLVM-LABEL: test_mm256_cvtneps_pbh
// LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
// OGCG-LABEL: test_mm256_cvtneps_pbh
// OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
__m128bh test_mm256_cvtneps_pbh(__m256 a) {
  return _mm256_cvtneps_pbh(a);
}

// CIR-LABEL: test_mm_cvtneps_pbh
// CIR: cir.call @_mm_cvtneps_pbh({{.*}})
// LLVM-LABEL: test_mm_cvtneps_pbh
// LLVM: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
// OGCG-LABEL: test_mm_cvtneps_pbh
// OGCG: call <4 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.128
__m64bh test_mm_cvtneps_pbh(__m128 a) {
  return _mm_cvtneps_pbh(a);
}

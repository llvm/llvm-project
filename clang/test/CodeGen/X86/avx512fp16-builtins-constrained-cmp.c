// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512fp16 -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__mmask32 test_mm512_cmp_round_ph_mask(__m512h a, __m512h b) {
  // CHECK-LABEL: @test_mm512_cmp_round_ph_mask
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 0, <32 x i1> {{.*}}, i32 8)
  return _mm512_cmp_round_ph_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask32 test_mm512_mask_cmp_round_ph_mask(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_ph_mask
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 0, <32 x i1> {{.*}}, i32 8)
  return _mm512_mask_cmp_round_ph_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_cmp_ph_mask_eq_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: @test_mm_cmp_ph_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 0, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_ph_mask_lt_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_lt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 1, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_ph_mask_le_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_le_os
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 2, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_ph_mask_unord_q(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_unord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 3, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_ph_mask_neq_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 4, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_ph_mask_nlt_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 5, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_ph_mask_nle_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nle_us
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 6, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_ph_mask_ord_q(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 7, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_ph_mask_eq_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 8, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_ph_mask_nge_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nge_us
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 9, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_ph_mask_ngt_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 10, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_ph_mask_false_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 11, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_ph_mask_neq_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 12, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_ph_mask_ge_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ge_os
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 13, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_ph_mask_gt_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_gt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 14, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_ph_mask_true_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 15, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_ph_mask_eq_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_eq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 16, <8 x i1> {{.*}})
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_OS);
}

__mmask16 test_mm256_cmp_ph_mask_lt_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_lt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 17, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_LT_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_le_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_le_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 18, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_LE_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_unord_s(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_unord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 19, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_UNORD_S);
}

__mmask16 test_mm256_cmp_ph_mask_neq_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 20, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_US);
}

__mmask16 test_mm256_cmp_ph_mask_nlt_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nlt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 21, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_nle_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nle_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 22, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_ord_s(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 23, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_ORD_S);
}

__mmask16 test_mm256_cmp_ph_mask_eq_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_eq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 24, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_EQ_US);
}

__mmask16 test_mm256_cmp_ph_mask_nge_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nge_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 25, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_ngt_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ngt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 26, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_false_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_false_os
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 27, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm256_cmp_ph_mask_neq_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_os
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 28, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm256_cmp_ph_mask_ge_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ge_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 29, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_GE_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_gt_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_gt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 30, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_GT_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_true_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_true_us
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 31, <16 x i1> {{.*}})
  return _mm256_cmp_ph_mask(a, b, _CMP_TRUE_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_eq_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ph_mask_eq_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 0, <16 x i1> {{.*}})
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_lt_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_lt_os
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 1, <16 x i1> {{.*}})
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_le_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_le_os
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 2, <16 x i1> {{.*}})
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LE_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_unord_q(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_unord_q
  // CHECK: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 3, <16 x i1> {{.*}})
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 4, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nlt_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nlt_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 5, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLT_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nle_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nle_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 6, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLE_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ord_q(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ord_q
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 7, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_ORD_Q);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 8, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nge_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nge_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 9, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGE_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ngt_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ngt_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 10, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGT_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_false_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_false_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 11, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 12, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ge_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ge_os
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 13, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GE_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_gt_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_gt_os
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 14, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GT_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_true_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_true_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 15, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_os
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 16, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_lt_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_lt_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 17, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LT_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_le_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_le_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 18, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_unord_s(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_unord_s
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 19, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_S);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 20, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nlt_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nlt_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 21, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nle_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nle_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 22, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ord_s(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ord_s
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 23, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_ORD_S);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 24, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nge_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nge_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 25, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ngt_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ngt_uq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 26, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_false_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_false_os
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 27, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_os
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 28, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ge_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ge_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 29, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_gt_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_gt_oq
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 30, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GT_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_true_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_true_us
  // CHECK: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 31, <32 x i1> {{.*}})
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_US);
}

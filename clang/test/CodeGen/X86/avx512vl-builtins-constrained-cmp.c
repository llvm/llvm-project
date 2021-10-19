// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -fexperimental-new-pass-manager -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512vl -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__mmask8 test_mm256_cmp_ps_mask_eq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_cmp_ps_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 0, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_lt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 1, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_le_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 2, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_unord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 3, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_neq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 4, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 5, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_ps_mask_nle_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 6, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 7, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_eq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nge_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_ps_mask_false_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_neq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 12, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_ge_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 13, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_gt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 14, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_true_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 15, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_eq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 16, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_lt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 17, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_le_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 18, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_unord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 19, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_neq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 20, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 21, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nle_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 22, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 23, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_eq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 24, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nge_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 25, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 26, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_false_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 27, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_neq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 28, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_ge_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 29, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_gt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 30, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_true_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 31, <8 x i1> {{.*}})
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 0, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 1, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 2, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 3, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 4, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 5, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 6, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 7, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 12, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 13, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 14, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 15, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 16, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 17, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 18, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 19, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 20, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 21, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 22, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 23, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 24, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 25, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 26, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 27, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 28, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 29, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 30, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 31, <8 x i1> {{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_cmp_pd_mask_eq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_cmp_pd_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 0, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_lt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 1, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_le_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 2, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_unord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 3, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_neq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 4, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 5, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_pd_mask_nle_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 6, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 7, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_eq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nge_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_pd_mask_false_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_neq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 12, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_ge_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 13, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_gt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 14, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_true_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 15, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_eq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 16, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_lt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 17, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_le_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 18, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_unord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 19, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_neq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 20, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 21, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nle_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 22, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 23, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_eq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 24, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nge_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 25, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 26, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_false_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 27, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_neq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 28, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_ge_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 29, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_gt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 30, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_true_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 31, <4 x i1> {{.*}})
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 0, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 1, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 2, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 3, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 4, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 5, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 6, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 7, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 12, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 13, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 14, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 15, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 16, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 17, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 18, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 19, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 20, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 21, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 22, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 23, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 24, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 25, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 26, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 27, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 28, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 29, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 30, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 31, <4 x i1> {{.*}})
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_ps_mask_eq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_cmp_ps_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_lt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_ps_mask_le_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_ps_mask_unord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 3, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_neq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 4, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nlt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 5, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_ps_mask_nle_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 6, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_ps_mask_ord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 7, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_eq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 8, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nge_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 9, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_ps_mask_ngt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 10, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_ps_mask_false_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 11, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_neq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_ge_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 13, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_ps_mask_gt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 14, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_ps_mask_true_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_eq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 16, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_lt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 17, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_le_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 18, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_unord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 19, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_ps_mask_neq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 20, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nlt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 21, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nle_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 22, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 23, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_ps_mask_eq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 24, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nge_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 25, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ngt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 26, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_false_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 27, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_ps_mask_neq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 28, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_ge_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 29, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_gt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 30, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_true_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 31, <4 x i1> {{.*}})
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_mask_cmp_ps_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 3, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 4, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 5, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 6, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_q
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 7, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 8, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 9, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 10, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 11, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 13, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 14, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 16, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 17, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 18, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 19, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 20, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 21, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 22, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_s
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 23, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 24, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 25, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 26, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 27, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 28, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 29, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 30, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 31, <4 x i1> {{.*}})
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_pd_mask_eq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_cmp_pd_mask_eq_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_lt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_pd_mask_le_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_pd_mask_unord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_q
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 3, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_neq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 4, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nlt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 5, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_pd_mask_nle_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 6, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_pd_mask_ord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_q
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 7, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_eq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 8, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nge_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 9, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_pd_mask_ngt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 10, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_pd_mask_false_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 11, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_neq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_ge_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 13, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_pd_mask_gt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 14, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_pd_mask_true_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_eq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 16, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_lt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 17, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_le_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 18, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_unord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_s
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 19, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_pd_mask_neq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 20, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nlt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 21, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nle_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 22, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_s
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 23, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_pd_mask_eq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 24, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nge_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 25, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ngt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 26, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_false_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 27, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_pd_mask_neq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 28, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_ge_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 29, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_gt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 30, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_true_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 31, <2 x i1> {{.*}})
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_cmp_pd_mask_eq_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_q
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 3, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 4, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 5, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 6, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_q
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 7, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 8, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 9, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 10, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 11, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 13, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 14, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 16, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 17, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 18, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_s
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 19, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 20, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 21, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 22, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_s
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 23, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 24, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 25, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 26, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 27, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 28, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 29, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 30, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 31, <2 x i1> {{.*}})
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

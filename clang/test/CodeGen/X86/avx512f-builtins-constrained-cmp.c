// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__mmask16 test_mm512_cmp_round_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_cmp_round_ps_mask
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 0, <16 x i1> {{.*}}, i32 8)
  return _mm512_cmp_round_ps_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm512_mask_cmp_round_ps_mask(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_ps_mask
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 0, <16 x i1> {{.*}}, i32 8)
  return _mm512_mask_cmp_round_ps_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm512_cmp_ps_mask_eq_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_cmp_ps_mask_eq_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 0, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_lt_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_lt_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 1, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm512_cmp_ps_mask_le_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_le_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 2, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_unord_q(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_unord_q
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 3, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm512_cmp_ps_mask_neq_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nlt_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nlt_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 5, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask16 test_mm512_cmp_ps_mask_nle_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nle_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 6, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask16 test_mm512_cmp_ps_mask_ord_q(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ord_q
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 7, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask16 test_mm512_cmp_ps_mask_eq_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 8, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nge_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nge_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 9, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask16 test_mm512_cmp_ps_mask_ngt_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ngt_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 10, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask16 test_mm512_cmp_ps_mask_false_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_false_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 11, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_neq_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 12, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_ge_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ge_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 13, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_gt_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_gt_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 14, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask16 test_mm512_cmp_ps_mask_true_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_true_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 15, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_eq_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 16, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask16 test_mm512_cmp_ps_mask_lt_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_lt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 17, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_le_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_le_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 18, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_unord_s(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_unord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 19, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask16 test_mm512_cmp_ps_mask_neq_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 20, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask16 test_mm512_cmp_ps_mask_nlt_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nlt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 21, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nle_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nle_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 22, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_ord_s(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 23, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask16 test_mm512_cmp_ps_mask_eq_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 24, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask16 test_mm512_cmp_ps_mask_nge_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nge_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 25, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_ngt_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ngt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 26, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_false_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_false_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 27, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_neq_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 28, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm512_cmp_ps_mask_ge_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ge_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 29, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_gt_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_gt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 30, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_true_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_true_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 31, <16 x i1> {{.*}}, i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_ps_mask_eq_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 0, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_lt_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_lt_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 1, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_le_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_le_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 2, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_unord_q(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_unord_q
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 3, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nlt_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nlt_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 5, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nle_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nle_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 6, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ord_q(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ord_q
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 7, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 8, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nge_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nge_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 9, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ngt_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ngt_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 10, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_false_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_false_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 11, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 12, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ge_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ge_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 13, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_gt_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_gt_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 14, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_true_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_true_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 15, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 16, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_lt_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_lt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 17, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_le_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_le_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 18, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_unord_s(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_unord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 19, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 20, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nlt_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nlt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 21, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nle_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nle_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 22, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ord_s(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ord_s
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 23, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 24, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nge_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nge_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 25, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ngt_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ngt_uq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 26, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_false_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_false_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 27, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_os
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 28, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ge_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ge_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 29, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_gt_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_gt_oq
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 30, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_true_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_true_us
  // CHECK: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 31, <16 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm512_cmp_round_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_cmp_round_pd_mask
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 0, <8 x i1> {{.*}}, i32 8)
  return _mm512_cmp_round_pd_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm512_mask_cmp_round_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_pd_mask
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 0, <8 x i1> {{.*}}, i32 8)
  return _mm512_mask_cmp_round_pd_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm512_cmp_pd_mask_eq_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_cmp_pd_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 0, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_lt_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_lt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 1, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm512_cmp_pd_mask_le_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_le_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 2, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_unord_q(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_unord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 3, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm512_cmp_pd_mask_neq_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nlt_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 5, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm512_cmp_pd_mask_nle_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nle_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 6, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm512_cmp_pd_mask_ord_q(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 7, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm512_cmp_pd_mask_eq_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 8, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nge_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nge_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 9, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm512_cmp_pd_mask_ngt_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 10, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm512_cmp_pd_mask_false_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 11, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_neq_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 12, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_ge_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ge_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 13, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_gt_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_gt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 14, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm512_cmp_pd_mask_true_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 15, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_eq_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 16, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm512_cmp_pd_mask_lt_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_lt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 17, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_le_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_le_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 18, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_unord_s(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_unord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 19, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm512_cmp_pd_mask_neq_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 20, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm512_cmp_pd_mask_nlt_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nlt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 21, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nle_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nle_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 22, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_ord_s(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 23, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm512_cmp_pd_mask_eq_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 24, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm512_cmp_pd_mask_nge_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nge_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 25, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_ngt_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ngt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 26, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_false_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_false_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 27, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_neq_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 28, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm512_cmp_pd_mask_ge_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ge_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 29, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_gt_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_gt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 30, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_true_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_true_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 31, <8 x i1> {{.*}}, i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_pd_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 0, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_lt_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_lt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 1, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_le_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_le_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 2, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_unord_q(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_unord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 3, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 5, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nle_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nle_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 6, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ord_q(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ord_q
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 7, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 8, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nge_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nge_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 9, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 10, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_false_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 11, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 12, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ge_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ge_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 13, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_gt_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_gt_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 14, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_true_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 15, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 16, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_lt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 17, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_le_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_le_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 18, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_unord_s(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_unord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 19, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 20, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nlt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 21, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nle_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 22, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ord_s(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ord_s
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 23, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 24, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nge_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 25, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ngt_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 26, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_false_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_false_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 27, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 28, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ge_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 29, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_gt_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 30, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_true_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_true_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 31, <8 x i1> {{.*}}, i32 4)
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -fexperimental-new-pass-manager -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512vl -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__mmask8 test_mm256_cmp_ps_mask_eq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_cmp_ps_mask_eq_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_lt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_le_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_unord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_q
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_neq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_ps_mask_nle_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_q
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_eq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nge_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_ps_mask_false_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_oq
  // CHECK: call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_neq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_ge_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_gt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_true_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_uq
  // CHECK: call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 15)
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_eq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_lt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_le_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_unord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_s
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_neq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nle_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_s
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_eq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_us
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nge_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_uq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_false_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_os
  // CHECK: call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 27)
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_neq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_os
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_ge_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_gt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_oq
  // CHECK: call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_true_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_us
  // CHECK: call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 31)
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_eq_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_q
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_q
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 15)
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_s
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_s
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 27)
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_os
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_oq
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_us
  // CHECK: [[CMP:%.*]] = call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 31)
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_cmp_pd_mask_eq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_cmp_pd_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_lt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_le_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_unord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_q
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_neq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_pd_mask_nle_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_q
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_eq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nge_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_pd_mask_false_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_neq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_ge_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_gt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_true_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 15)
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_eq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_lt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_le_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_unord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_s
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_neq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nle_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_s
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_eq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nge_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_false_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 27)
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_neq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_ge_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_gt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_true_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 31)
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask_eq_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_q
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_q
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 15)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_s
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_s
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 27)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 31)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_ps_mask_eq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_cmp_ps_mask_eq_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_lt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_ps_mask_le_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_ps_mask_unord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_q
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_neq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nlt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_ps_mask_nle_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_ps_mask_ord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_q
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_eq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nge_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_ps_mask_ngt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_ps_mask_false_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_oq
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 11)
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_neq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_ge_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_ps_mask_gt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_ps_mask_true_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_uq
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_eq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_lt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_le_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_unord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_s
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_ps_mask_neq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nlt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nle_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_s
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_ps_mask_eq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_us
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nge_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ngt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_uq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_false_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_os
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 27)
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_ps_mask_neq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_os
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_ge_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_gt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_oq
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_true_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_us
  // CHECK: call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 31)
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_mask_cmp_ps_mask_eq_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_q
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_q
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 11)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_s
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_s
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 27)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_os
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_oq
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_us
  // CHECK: [[CMP:%.*]] = call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 31)
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_pd_mask_eq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_cmp_pd_mask_eq_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_lt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_pd_mask_le_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_pd_mask_unord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_q
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_neq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nlt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_pd_mask_nle_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_pd_mask_ord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_q
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_eq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nge_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_pd_mask_ngt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_pd_mask_false_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_oq
  // CHECK: call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 11)
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_neq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_ge_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_pd_mask_gt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_pd_mask_true_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_uq
  // CHECK: call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15)
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_eq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_lt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_le_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_unord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_s
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_pd_mask_neq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nlt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nle_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_s
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_pd_mask_eq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_us
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nge_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ngt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_uq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_false_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_os
  // CHECK: call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 27)
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_pd_mask_neq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_os
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_ge_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_gt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_oq
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_true_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_us
  // CHECK: call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 31)
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_cmp_pd_mask_eq_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_q
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_q
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_oq
  // [[CMP]] = call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 11)
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_uq
  // [[CMP]] = call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15)
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_s
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_s
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_us
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_os
  // [[CMP]] = call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 27)
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_os
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_oq
  // CHECK: [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_us
  // [[CMP]] = call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 31)
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

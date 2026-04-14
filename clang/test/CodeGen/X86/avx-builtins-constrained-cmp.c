// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -ffp-exception-behavior=maytrap -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -emit-llvm -ffp-exception-behavior=maytrap -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -ffp-exception-behavior=maytrap -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -emit-llvm -ffp-exception-behavior=maytrap -o - -Wall -Werror | FileCheck %s

// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)


#include <immintrin.h>

__m256d test_mm256_cmp_pd_eq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq")
  return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}

__m256d test_mm256_cmp_pd_lt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_lt_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt")
  return _mm256_cmp_pd(a, b, _CMP_LT_OS);
}

__m256d test_mm256_cmp_pd_le_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_le_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole")
  return _mm256_cmp_pd(a, b, _CMP_LE_OS);
}

__m256d test_mm256_cmp_pd_unord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_unord_q
  // CHECK: all <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno")
  return _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
}

__m256d test_mm256_cmp_pd_neq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une")
  return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ);
}

__m256d test_mm256_cmp_pd_nlt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nlt_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge")
  return _mm256_cmp_pd(a, b, _CMP_NLT_US);
}

__m256d test_mm256_cmp_pd_nle_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nle_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt")
  return _mm256_cmp_pd(a, b, _CMP_NLE_US);
}

__m256d test_mm256_cmp_pd_ord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ord_q
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord")
  return _mm256_cmp_pd(a, b, _CMP_ORD_Q);
}

__m256d test_mm256_cmp_pd_eq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq")
  return _mm256_cmp_pd(a, b, _CMP_EQ_UQ);
}

__m256d test_mm256_cmp_pd_nge_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nge_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult")
  return _mm256_cmp_pd(a, b, _CMP_NGE_US);
}

__m256d test_mm256_cmp_pd_ngt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ngt_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule")
  return _mm256_cmp_pd(a, b, _CMP_NGT_US);
}

__m256d test_mm256_cmp_pd_false_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_false_oq
  // CHECK: call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i8 11)
  return _mm256_cmp_pd(a, b, _CMP_FALSE_OQ);
}

__m256d test_mm256_cmp_pd_neq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one")
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
}

__m256d test_mm256_cmp_pd_ge_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ge_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge")
  return _mm256_cmp_pd(a, b, _CMP_GE_OS);
}

__m256d test_mm256_cmp_pd_gt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_gt_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt")
  return _mm256_cmp_pd(a, b, _CMP_GT_OS);
}

__m256d test_mm256_cmp_pd_true_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_true_uq
  // CHECK: call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i8 15)
  return _mm256_cmp_pd(a, b, _CMP_TRUE_UQ);
}

__m256d test_mm256_cmp_pd_eq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oeq")
  return _mm256_cmp_pd(a, b, _CMP_EQ_OS);
}

__m256d test_mm256_cmp_pd_lt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_lt_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt")
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

__m256d test_mm256_cmp_pd_le_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_le_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ole")
  return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
}

__m256d test_mm256_cmp_pd_unord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_unord_s
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uno")
  return _mm256_cmp_pd(a, b, _CMP_UNORD_S);
}

__m256d test_mm256_cmp_pd_neq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"une")
  return _mm256_cmp_pd(a, b, _CMP_NEQ_US);
}

__m256d test_mm256_cmp_pd_nlt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nlt_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"uge")
  return _mm256_cmp_pd(a, b, _CMP_NLT_UQ);
}

__m256d test_mm256_cmp_pd_nle_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nle_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ugt")
  return _mm256_cmp_pd(a, b, _CMP_NLE_UQ);
}

__m256d test_mm256_cmp_pd_ord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ord_s
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ord")
  return _mm256_cmp_pd(a, b, _CMP_ORD_S);
}

__m256d test_mm256_cmp_pd_eq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ueq")
  return _mm256_cmp_pd(a, b, _CMP_EQ_US);
}

__m256d test_mm256_cmp_pd_nge_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nge_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ult")
  return _mm256_cmp_pd(a, b, _CMP_NGE_UQ);
}

__m256d test_mm256_cmp_pd_ngt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ngt_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ule")
  return _mm256_cmp_pd(a, b, _CMP_NGT_UQ);
}

__m256d test_mm256_cmp_pd_false_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_false_os
  // CHECK: call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i8 27)
  return _mm256_cmp_pd(a, b, _CMP_FALSE_OS);
}

__m256d test_mm256_cmp_pd_neq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"one")
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OS);
}

__m256d test_mm256_cmp_pd_ge_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ge_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"oge")
  return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}

__m256d test_mm256_cmp_pd_gt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_gt_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"ogt")
  return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
}

__m256d test_mm256_cmp_pd_true_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_true_us
  // CHECK: call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i8 31)
  return _mm256_cmp_pd(a, b, _CMP_TRUE_US);
}

__m256 test_mm256_cmp_ps_eq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq")
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

__m256 test_mm256_cmp_ps_lt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_lt_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt")
  return _mm256_cmp_ps(a, b, _CMP_LT_OS);
}

__m256 test_mm256_cmp_ps_le_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_le_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole")
  return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

__m256 test_mm256_cmp_ps_unord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_unord_q
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno")
  return _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
}

__m256 test_mm256_cmp_ps_neq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une")
  return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
}

__m256 test_mm256_cmp_ps_nlt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nlt_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge")
  return _mm256_cmp_ps(a, b, _CMP_NLT_US);
}

__m256 test_mm256_cmp_ps_nle_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nle_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt")
  return _mm256_cmp_ps(a, b, _CMP_NLE_US);
}

__m256 test_mm256_cmp_ps_ord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ord_q
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord")
  return _mm256_cmp_ps(a, b, _CMP_ORD_Q);
}

__m256 test_mm256_cmp_ps_eq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq")
  return _mm256_cmp_ps(a, b, _CMP_EQ_UQ);
}

__m256 test_mm256_cmp_ps_nge_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nge_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult")
  return _mm256_cmp_ps(a, b, _CMP_NGE_US);
}

__m256 test_mm256_cmp_ps_ngt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ngt_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule")
  return _mm256_cmp_ps(a, b, _CMP_NGT_US);
}

__m256 test_mm256_cmp_ps_false_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_false_oq
  // CHECK: call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i8 11)
  return _mm256_cmp_ps(a, b, _CMP_FALSE_OQ);
}

__m256 test_mm256_cmp_ps_neq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one")
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}

__m256 test_mm256_cmp_ps_ge_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ge_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge")
  return _mm256_cmp_ps(a, b, _CMP_GE_OS);
}

__m256 test_mm256_cmp_ps_gt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_gt_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt")
  return _mm256_cmp_ps(a, b, _CMP_GT_OS);
}

__m256 test_mm256_cmp_ps_true_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_true_uq
  // CHECK: call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i8 15)
  return _mm256_cmp_ps(a, b, _CMP_TRUE_UQ);
}

__m256 test_mm256_cmp_ps_eq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oeq")
  return _mm256_cmp_ps(a, b, _CMP_EQ_OS);
}

__m256 test_mm256_cmp_ps_lt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_lt_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"olt")
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

__m256 test_mm256_cmp_ps_le_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_le_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ole")
  return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

__m256 test_mm256_cmp_ps_unord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_unord_s
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uno")
  return _mm256_cmp_ps(a, b, _CMP_UNORD_S);
}

__m256 test_mm256_cmp_ps_neq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"une")
  return _mm256_cmp_ps(a, b, _CMP_NEQ_US);
}

__m256 test_mm256_cmp_ps_nlt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nlt_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"uge")
  return _mm256_cmp_ps(a, b, _CMP_NLT_UQ);
}

__m256 test_mm256_cmp_ps_nle_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nle_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ugt")
  return _mm256_cmp_ps(a, b, _CMP_NLE_UQ);
}

__m256 test_mm256_cmp_ps_ord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ord_s
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ord")
  return _mm256_cmp_ps(a, b, _CMP_ORD_S);
}

__m256 test_mm256_cmp_ps_eq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_us
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ueq")
  return _mm256_cmp_ps(a, b, _CMP_EQ_US);
}

__m256 test_mm256_cmp_ps_nge_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nge_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ult")
  return _mm256_cmp_ps(a, b, _CMP_NGE_UQ);
}

__m256 test_mm256_cmp_ps_ngt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ngt_uq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ule")
  return _mm256_cmp_ps(a, b, _CMP_NGT_UQ);
}

__m256 test_mm256_cmp_ps_false_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_false_os
  // CHECK: call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i8 27)
  return _mm256_cmp_ps(a, b, _CMP_FALSE_OS);
}

__m256 test_mm256_cmp_ps_neq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_os
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"one")
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OS);
}

__m256 test_mm256_cmp_ps_ge_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ge_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"oge")
  return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

__m256 test_mm256_cmp_ps_gt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_gt_oq
  // CHECK: call <8 x i1> @llvm.fcmp.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !"ogt")
  return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

__m256 test_mm256_cmp_ps_true_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_true_us
  // CHECK: call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i8 31)
  return _mm256_cmp_ps(a, b, _CMP_TRUE_US);
}

__m128d test_mm_cmp_pd_eq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq")
  return _mm_cmp_pd(a, b, _CMP_EQ_OQ);
}

__m128d test_mm_cmp_pd_lt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_lt_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt")
  return _mm_cmp_pd(a, b, _CMP_LT_OS);
}

__m128d test_mm_cmp_pd_le_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_le_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole")
  return _mm_cmp_pd(a, b, _CMP_LE_OS);
}

__m128d test_mm_cmp_pd_unord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_unord_q
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno")
  return _mm_cmp_pd(a, b, _CMP_UNORD_Q);
}

__m128d test_mm_cmp_pd_neq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une")
  return _mm_cmp_pd(a, b, _CMP_NEQ_UQ);
}

__m128d test_mm_cmp_pd_nlt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nlt_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge")
  return _mm_cmp_pd(a, b, _CMP_NLT_US);
}

__m128d test_mm_cmp_pd_nle_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nle_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt")
  return _mm_cmp_pd(a, b, _CMP_NLE_US);
}

__m128d test_mm_cmp_pd_ord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ord_q
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord")
  return _mm_cmp_pd(a, b, _CMP_ORD_Q);
}

__m128d test_mm_cmp_pd_eq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq")
  return _mm_cmp_pd(a, b, _CMP_EQ_UQ);
}

__m128d test_mm_cmp_pd_nge_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nge_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult")
  return _mm_cmp_pd(a, b, _CMP_NGE_US);
}

__m128d test_mm_cmp_pd_ngt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ngt_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule")
  return _mm_cmp_pd(a, b, _CMP_NGT_US);
}

__m128d test_mm_cmp_pd_false_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_false_oq
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 11)
  return _mm_cmp_pd(a, b, _CMP_FALSE_OQ);
}

__m128d test_mm_cmp_pd_neq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one")
  return _mm_cmp_pd(a, b, _CMP_NEQ_OQ);
}

__m128d test_mm_cmp_pd_ge_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ge_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge")
  return _mm_cmp_pd(a, b, _CMP_GE_OS);
}

__m128d test_mm_cmp_pd_gt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_gt_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt")
  return _mm_cmp_pd(a, b, _CMP_GT_OS);
}

__m128d test_mm_cmp_pd_true_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_true_uq
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 15)
  return _mm_cmp_pd(a, b, _CMP_TRUE_UQ);
}

__m128d test_mm_cmp_pd_eq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq")
  return _mm_cmp_pd(a, b, _CMP_EQ_OS);
}

__m128d test_mm_cmp_pd_lt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_lt_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt")
  return _mm_cmp_pd(a, b, _CMP_LT_OQ);
}

__m128d test_mm_cmp_pd_le_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_le_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole")
  return _mm_cmp_pd(a, b, _CMP_LE_OQ);
}

__m128d test_mm_cmp_pd_unord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_unord_s
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno")
  return _mm_cmp_pd(a, b, _CMP_UNORD_S);
}

__m128d test_mm_cmp_pd_neq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une")
  return _mm_cmp_pd(a, b, _CMP_NEQ_US);
}

__m128d test_mm_cmp_pd_nlt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nlt_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge")
  return _mm_cmp_pd(a, b, _CMP_NLT_UQ);
}

__m128d test_mm_cmp_pd_nle_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nle_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt")
  return _mm_cmp_pd(a, b, _CMP_NLE_UQ);
}

__m128d test_mm_cmp_pd_ord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ord_s
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord")
  return _mm_cmp_pd(a, b, _CMP_ORD_S);
}

__m128d test_mm_cmp_pd_eq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_us
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ueq")
  return _mm_cmp_pd(a, b, _CMP_EQ_US);
}

__m128d test_mm_cmp_pd_nge_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nge_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ult")
  return _mm_cmp_pd(a, b, _CMP_NGE_UQ);
}

__m128d test_mm_cmp_pd_ngt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ngt_uq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ule")
  return _mm_cmp_pd(a, b, _CMP_NGT_UQ);
}

__m128d test_mm_cmp_pd_false_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_false_os
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 27)
  return _mm_cmp_pd(a, b, _CMP_FALSE_OS);
}

__m128d test_mm_cmp_pd_neq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_os
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"one")
  return _mm_cmp_pd(a, b, _CMP_NEQ_OS);
}

__m128d test_mm_cmp_pd_ge_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ge_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge")
  return _mm_cmp_pd(a, b, _CMP_GE_OQ);
}

__m128d test_mm_cmp_pd_gt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_gt_oq
  // CHECK: call <2 x i1> @llvm.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt")
  return _mm_cmp_pd(a, b, _CMP_GT_OQ);
}

__m128d test_mm_cmp_pd_true_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_true_us
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 31)
  return _mm_cmp_pd(a, b, _CMP_TRUE_US);
}

__m128 test_mm_cmp_ps_eq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq")
  return _mm_cmp_ps(a, b, _CMP_EQ_OQ);
}

__m128 test_mm_cmp_ps_lt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_lt_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt")
  return _mm_cmp_ps(a, b, _CMP_LT_OS);
}

__m128 test_mm_cmp_ps_le_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_le_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole")
  return _mm_cmp_ps(a, b, _CMP_LE_OS);
}

__m128 test_mm_cmp_ps_unord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_unord_q
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno")
  return _mm_cmp_ps(a, b, _CMP_UNORD_Q);
}

__m128 test_mm_cmp_ps_neq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une")
  return _mm_cmp_ps(a, b, _CMP_NEQ_UQ);
}

__m128 test_mm_cmp_ps_nlt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nlt_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge")
  return _mm_cmp_ps(a, b, _CMP_NLT_US);
}

__m128 test_mm_cmp_ps_nle_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nle_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt")
  return _mm_cmp_ps(a, b, _CMP_NLE_US);
}

__m128 test_mm_cmp_ps_ord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ord_q
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord")
  return _mm_cmp_ps(a, b, _CMP_ORD_Q);
}

__m128 test_mm_cmp_ps_eq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq")
  return _mm_cmp_ps(a, b, _CMP_EQ_UQ);
}

__m128 test_mm_cmp_ps_nge_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nge_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult")
  return _mm_cmp_ps(a, b, _CMP_NGE_US);
}

__m128 test_mm_cmp_ps_ngt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ngt_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule")
  return _mm_cmp_ps(a, b, _CMP_NGT_US);
}

__m128 test_mm_cmp_ps_false_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_false_oq
  // CHECK: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 11)
  return _mm_cmp_ps(a, b, _CMP_FALSE_OQ);
}

__m128 test_mm_cmp_ps_neq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one")
  return _mm_cmp_ps(a, b, _CMP_NEQ_OQ);
}

__m128 test_mm_cmp_ps_ge_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ge_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge")
  return _mm_cmp_ps(a, b, _CMP_GE_OS);
}

__m128 test_mm_cmp_ps_gt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_gt_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt")
  return _mm_cmp_ps(a, b, _CMP_GT_OS);
}

__m128 test_mm_cmp_ps_true_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_true_uq
  // CHECK: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 15)
  return _mm_cmp_ps(a, b, _CMP_TRUE_UQ);
}

__m128 test_mm_cmp_ps_eq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq")
  return _mm_cmp_ps(a, b, _CMP_EQ_OS);
}

__m128 test_mm_cmp_ps_lt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_lt_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt")
  return _mm_cmp_ps(a, b, _CMP_LT_OQ);
}

__m128 test_mm_cmp_ps_le_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_le_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole")
  return _mm_cmp_ps(a, b, _CMP_LE_OQ);
}

__m128 test_mm_cmp_ps_unord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_unord_s
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno")
  return _mm_cmp_ps(a, b, _CMP_UNORD_S);
}

__m128 test_mm_cmp_ps_neq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une")
  return _mm_cmp_ps(a, b, _CMP_NEQ_US);
}

__m128 test_mm_cmp_ps_nlt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nlt_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge")
  return _mm_cmp_ps(a, b, _CMP_NLT_UQ);
}

__m128 test_mm_cmp_ps_nle_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nle_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt")
  return _mm_cmp_ps(a, b, _CMP_NLE_UQ);
}

__m128 test_mm_cmp_ps_ord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ord_s
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord")
  return _mm_cmp_ps(a, b, _CMP_ORD_S);
}

__m128 test_mm_cmp_ps_eq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_us
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ueq")
  return _mm_cmp_ps(a, b, _CMP_EQ_US);
}

__m128 test_mm_cmp_ps_nge_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nge_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult")
  return _mm_cmp_ps(a, b, _CMP_NGE_UQ);
}

__m128 test_mm_cmp_ps_ngt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ngt_uq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ule")
  return _mm_cmp_ps(a, b, _CMP_NGT_UQ);
}

__m128 test_mm_cmp_ps_false_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_false_os
  // CHECK: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 27)
  return _mm_cmp_ps(a, b, _CMP_FALSE_OS);
}

__m128 test_mm_cmp_ps_neq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_os
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"one")
  return _mm_cmp_ps(a, b, _CMP_NEQ_OS);
}

__m128 test_mm_cmp_ps_ge_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ge_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge")
  return _mm_cmp_ps(a, b, _CMP_GE_OQ);
}

__m128 test_mm_cmp_ps_gt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_gt_oq
  // CHECK: call <4 x i1> @llvm.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt")
  return _mm_cmp_ps(a, b, _CMP_GT_OQ);
}

__m128 test_mm_cmp_ps_true_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_true_us
  // CHECK: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 31)
  return _mm_cmp_ps(a, b, _CMP_TRUE_US);
}

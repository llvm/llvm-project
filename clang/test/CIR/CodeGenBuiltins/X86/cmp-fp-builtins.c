// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=CHECK,LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,OGCG

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -ffp-exception-behavior=strict -fexperimental-strict-floating-point -fclangir -emit-cir -o %t.strict.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-STRICT --input-file=%t.strict.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -ffp-exception-behavior=strict -fexperimental-strict-floating-point -fclangir -emit-llvm -o %t.strict.ll -Wall -Werror
// RUN: FileCheck --check-prefix=CHECK-STRICT --input-file=%t.strict.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-STRICT

#include <immintrin.h>

__m128 test_cmp_ps_eq_oq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_eq_oq
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(oeq, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_eq_oq
  // CIR-STRICT: cir.vec.fcmp(oeq, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>

  // CHECK-LABEL: @test_cmp_ps_eq_oq
  // CHECK: %[[CMP:.*]] = fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: %[[SEXT:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
  // CHECK: bitcast <4 x i32> %[[SEXT]] to <4 x float>

  // CHECK-STRICT-LABEL: @test_cmp_ps_eq_oq
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_EQ_OQ);
}

// EQ_OS is signaling although eq is quiet by default.
__m128 test_cmp_ps_eq_os(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_eq_os
  // CIR-STRICT: cir.vec.fcmp(oeq, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>

  // CHECK-STRICT-LABEL: @test_cmp_ps_eq_os
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_EQ_OS);
}

// LT_OQ is quiet although lt is signaling by default.
__m256d test_cmp_pd_lt_oq(__m256d a, __m256d b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_pd_lt_oq
  // CIR-STRICT: cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.double>, !cir.vector<4 x !s64i> fenv<{{.*}}strict_except = true>

  // CHECK-STRICT-LABEL: @test_cmp_pd_lt_oq
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

__m128 test_cmp_ps_nlt_us(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_nlt_us
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(uge, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_nlt_us
  // CHECK: %[[CMP:.*]] = fcmp uge <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>

  // CHECK-STRICT-LABEL: @test_cmp_ps_nlt_us
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_NLT_US);
}

// FALSE and TRUE
__m128 test_cmp_ps_false_oq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_false_oq
  // CIR: %[[CMP:.*]] = cir.const #cir.const_vector<[#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_false_oq
  // LLVM: store <4 x float> zeroinitializer
  // OGCG: fcmp false <4 x float> %{{.*}}, %{{.*}}

  // Under strict FP there is no compare to use, so the target intrinsic is
  // called.
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_false_oq
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.sse.cmp.ps"

  // CHECK-STRICT-LABEL: @test_cmp_ps_false_oq
  // CHECK-STRICT: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 11)
  return _mm_cmp_ps(a, b, _CMP_FALSE_OQ);
}

__m128 test_cmp_ps_true_uq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_true_uq
  // CIR: %[[CMP:.*]] = cir.const #cir.const_vector<[#cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i]> : !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_true_uq
  // LLVM: store <4 x float> splat (float {{.*}})
  // OGCG: fcmp true <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_TRUE_UQ);
}

// Masked compares.
__mmask8 test_mm256_mask_cmp_ps_mask(__mmask8 m, __m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_ps_mask
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.float>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.and

  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: and <8 x i1> %[[CMP]], %[[MASK]]

  // CHECK-STRICT-LABEL: @test_mm256_mask_cmp_ps_mask
  // CHECK-STRICT: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

// Remaining non-signaling-flipped predicates.
__m128 test_cmp_ps_lt_os(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_lt_os
  // CIR: cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_lt_os
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_LT_OS);
}

__m128 test_cmp_ps_le_os(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_le_os
  // CIR: cir.vec.fcmp(ole, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_le_os
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_LE_OS);
}

__m128 test_cmp_ps_unord_q(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_unord_q
  // CIR: cir.vec.fcmp(uno, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_unord_q
  // CHECK: fcmp uno <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_UNORD_Q);
}

__m128 test_cmp_ps_neq_uq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_neq_uq
  // CIR: cir.vec.fcmp(une, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_neq_uq
  // CHECK: fcmp une <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NEQ_UQ);
}

__m128 test_cmp_ps_nle_us(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_nle_us
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(ugt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_nle_us
  // CHECK: %[[CMP:.*]] = fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>
  return _mm_cmp_ps(a, b, _CMP_NLE_US);
}

__m128 test_cmp_ps_ord_q(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_ord_q
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(ord, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_ord_q
  // CHECK: %[[CMP:.*]] = fcmp ord <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>
  return _mm_cmp_ps(a, b, _CMP_ORD_Q);
}

__m128 test_cmp_ps_eq_uq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_eq_uq
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(ueq, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_eq_uq
  // CHECK: %[[CMP:.*]] = fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>
  return _mm_cmp_ps(a, b, _CMP_EQ_UQ);
}

__m128 test_cmp_ps_nge_us(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_nge_us
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(ult, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_nge_us
  // CHECK: %[[CMP:.*]] = fcmp ult <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>
  return _mm_cmp_ps(a, b, _CMP_NGE_US);
}

__m128 test_cmp_ps_ngt_us(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_ngt_us
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(ule, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps_ngt_us
  // CHECK: %[[CMP:.*]] = fcmp ule <4 x float> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %[[CMP]] to <4 x i32>
  return _mm_cmp_ps(a, b, _CMP_NGT_US);
}

__m128 test_cmp_ps_neq_oq(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_neq_oq
  // CIR: cir.vec.fcmp(one, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_neq_oq
  // CHECK: fcmp one <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NEQ_OQ);
}

__m128 test_cmp_ps_ge_os(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_ge_os
  // CIR: cir.vec.fcmp(oge, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_ge_os
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GE_OS);
}

__m128 test_cmp_ps_gt_os(__m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps_gt_os
  // CIR: cir.vec.fcmp(ogt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

  // CHECK-LABEL: @test_cmp_ps_gt_os
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GT_OS);
}

// Predicates 16-31 flip the signaling behavior of predicates 0-15.
__m128 test_cmp_ps_neq_us(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_neq_us
  // CIR-STRICT: cir.vec.fcmp(une, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>

  // CHECK-STRICT-LABEL: @test_cmp_ps_neq_us
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_NEQ_US);
}

__m128 test_cmp_ps_unord_s(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_unord_s
  // CIR-STRICT: cir.vec.fcmp(uno, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>

  // CHECK-STRICT-LABEL: @test_cmp_ps_unord_s
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_UNORD_S);
}

__m128 test_cmp_ps_ge_oq(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_ge_oq
  // CIR-STRICT: cir.vec.fcmp(oge, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>

  // CHECK-STRICT-LABEL: @test_cmp_ps_ge_oq
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_GE_OQ);
}

__m128 test_cmp_ps_nge_uq(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_nge_uq
  // CIR-STRICT: %[[CMP:.*]] = cir.vec.fcmp(ult, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>
  // CIR-STRICT: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-STRICT-LABEL: @test_cmp_ps_nge_uq
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ult", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_NGE_UQ);
}

__m128 test_cmp_ps_ord_s(__m128 a, __m128 b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_ps_ord_s
  // CIR-STRICT: %[[CMP:.*]] = cir.vec.fcmp(ord, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>
  // CIR-STRICT: cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // CHECK-STRICT-LABEL: @test_cmp_ps_ord_s
  // CHECK-STRICT: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return _mm_cmp_ps(a, b, _CMP_ORD_S);
}

__m128d test_cmp_pd_lt_os(__m128d a, __m128d b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_pd_lt_os
  // CIR: cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !cir.double>

  // CHECK-LABEL: @test_cmp_pd_lt_os
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_LT_OS);
}

__m256 test_cmp_ps256_le_os(__m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_cmp_ps256_le_os
  // CIR: cir.vec.fcmp(ole, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.float>, !cir.vector<8 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s32i> -> !cir.vector<8 x !cir.float>

  // CHECK-LABEL: @test_cmp_ps256_le_os
  // CHECK: fcmp ole <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

__m128d test_cmp_pd_true_us_strict(__m128d a, __m128d b) {
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_cmp_pd_true_us_strict
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.sse2.cmp.pd"

  // CHECK-STRICT-LABEL: @test_cmp_pd_true_us_strict
  // CHECK-STRICT: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 31)
  return _mm_cmp_pd(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask(__mmask8 m, __m128 a, __m128 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm_mask_cmp_ps_mask
  // CIR: cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.and
  // CIR: cir.vec.shuffle

  // CHECK-LABEL: @test_mm_mask_cmp_ps_mask
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1>
  // CHECK: shufflevector <4 x i1>
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask(__mmask8 m, __m128d a, __m128d b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm_mask_cmp_pd_mask
  // CIR: cir.vec.fcmp(ole, %{{.*}}, %{{.*}}) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.int<s, 1>>
  // CIR: cir.and

  // CHECK-LABEL: @test_mm_mask_cmp_pd_mask
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1>
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask(__mmask8 m, __m256d a, __m256d b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_pd_mask
  // CIR: cir.vec.fcmp(ogt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.double>, !cir.vector<4 x !cir.int<s, 1>>
  // CIR: cir.and

  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask
  // CHECK: fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1>
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_ps_mask(__m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_cmp_ps_mask
  // CIR: cir.vec.fcmp(olt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.float>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm256_cmp_ps_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: bitcast <8 x i1> %[[CMP]] to i8
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false(__mmask8 m, __m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_ps_mask_false
  // CIR: %[[CMP:.*]] = cir.const #cir.const_vector<[#cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>, #cir.int<0> : !cir.int<s, 1>]> : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.and %{{.*}}, %[[CMP]]

  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_false
  // LLVM: and <8 x i1> %{{.*}}, zeroinitializer
  // OGCG: %[[CMP:.*]] = fcmp false <8 x float> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %[[CMP]], %{{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true(__mmask8 m, __m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_ps_mask_true
  // CIR: %[[CMP:.*]] = cir.const #cir.const_vector<[#cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>, #cir.int<-1> : !cir.int<s, 1>]> : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.and %{{.*}}, %[[CMP]]

  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_true
  // LLVM: and <8 x i1> %{{.*}}, splat (i1 true)
  // OGCG: %[[CMP:.*]] = fcmp true <8 x float> %{{.*}}, %{{.*}}
  // OGCG: and <8 x i1> %[[CMP]], %{{.*}}

  // Under strict FP the target intrinsic is called.
  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_ps_mask_true
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512.mask.cmp.ps.256"

  // CHECK-STRICT-LABEL: @test_mm256_mask_cmp_ps_mask_true
  // CHECK-STRICT: call <8 x i1> @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 15, <8 x i1> %{{.*}})
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m256 a, __m256 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_mask_cmp_ps_mask_nlt_us
  // CIR: %[[CMP:.*]] = cir.vec.fcmp(uge, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.float>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.and %[[CMP]], %{{.*}}

  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_nlt_us
  // CHECK: %[[CMP:.*]] = fcmp uge <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %[[CMP]], %{{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

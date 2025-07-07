// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64


#include <immintrin.h>
#include "builtin_test_helpers.h"

// NOTE: This should match the tests in llvm/test/CodeGen/X86/avx-intrinsics-fast-isel.ll

__m256d test_mm256_add_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_add_pd
  // CHECK: fadd <4 x double>
  return _mm256_add_pd(A, B);
}

__m256 test_mm256_add_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_add_ps
  // CHECK: fadd <8 x float>
  return _mm256_add_ps(A, B);
}

__m256d test_mm256_addsub_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_addsub_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_addsub_pd(A, B);
}

__m256 test_mm256_addsub_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_addsub_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_addsub_ps(A, B);
}

__m256d test_mm256_and_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_and_pd
  // CHECK: and <4 x i64>
  return _mm256_and_pd(A, B);
}

__m256 test_mm256_and_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_and_ps
  // CHECK: and <8 x i32>
  return _mm256_and_ps(A, B);
}

__m256d test_mm256_andnot_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_andnot_pd
  // CHECK: xor <4 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <4 x i64>
  return _mm256_andnot_pd(A, B);
}

__m256 test_mm256_andnot_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_andnot_ps
  // CHECK: xor <8 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <8 x i32>
  return _mm256_andnot_ps(A, B);
}

__m256d test_mm256_blend_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_blend_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm256_blend_pd(A, B, 0x05);
}

__m256 test_mm256_blend_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_blend_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_ps(A, B, 0x35);
}

__m256d test_mm256_blendv_pd(__m256d V1, __m256d V2, __m256d V3) {
  // CHECK-LABEL: test_mm256_blendv_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_blendv_pd(V1, V2, V3);
}

__m256 test_mm256_blendv_ps(__m256 V1, __m256 V2, __m256 V3) {
  // CHECK-LABEL: test_mm256_blendv_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_blendv_ps(V1, V2, V3);
}

__m256d test_mm256_broadcast_pd(__m128d* A) {
  // CHECK-LABEL: test_mm256_broadcast_pd
  // CHECK: load <2 x double>, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcast_pd(A);
}

__m256 test_mm256_broadcast_ps(__m128* A) {
  // CHECK-LABEL: test_mm256_broadcast_ps
  // CHECK: load <4 x float>, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm256_broadcast_ps(A);
}

__m256d test_mm256_broadcast_sd(double* A) {
  // CHECK-LABEL: test_mm256_broadcast_sd
  // CHECK: load double, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <4 x double> poison, double %{{.*}}, i32 0
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 1
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 2
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 3
  return _mm256_broadcast_sd(A);
}

__m128 test_mm_broadcast_ss(float* A) {
  // CHECK-LABEL: test_mm_broadcast_ss
  // CHECK: load float, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <4 x float> poison, float %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 3
  return _mm_broadcast_ss(A);
}

__m256 test_mm256_broadcast_ss(float* A) {
  // CHECK-LABEL: test_mm256_broadcast_ss
  // CHECK: load float, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <8 x float> poison, float %{{.*}}, i32 0
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 3
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 4
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 5
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 6
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 7
  return _mm256_broadcast_ss(A);
}

__m256 test_mm256_castpd_ps(__m256d A) {
  // CHECK-LABEL: test_mm256_castpd_ps
  return _mm256_castpd_ps(A);
}

__m256i test_mm256_castpd_si256(__m256d A) {
  // CHECK-LABEL: test_mm256_castpd_si256
  return _mm256_castpd_si256(A);
}

__m256d test_mm256_castpd128_pd256(__m128d A) {
  // CHECK-LABEL: test_mm256_castpd128_pd256
  // CHECK: [[A:%.*]] = freeze <2 x double> poison
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_castpd128_pd256(A);
}

__m128d test_mm256_castpd256_pd128(__m256d A) {
  // CHECK-LABEL: test_mm256_castpd256_pd128
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <2 x i32> <i32 0, i32 1>
  return _mm256_castpd256_pd128(A);
}

__m256d test_mm256_castps_pd(__m256 A) {
  // CHECK-LABEL: test_mm256_castps_pd
  return _mm256_castps_pd(A);
}

__m256i test_mm256_castps_si256(__m256 A) {
  // CHECK-LABEL: test_mm256_castps_si256
  return _mm256_castps_si256(A);
}

__m256 test_mm256_castps128_ps256(__m128 A) {
  // CHECK-LABEL: test_mm256_castps128_ps256
  // CHECK: [[A:%.*]] = freeze <4 x float> poison
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> [[A]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_castps128_ps256(A);
}

__m128 test_mm256_castps256_ps128(__m256 A) {
  // CHECK-LABEL: test_mm256_castps256_ps128
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_castps256_ps128(A);
}

__m256i test_mm256_castsi128_si256(__m128i A) {
  // CHECK-LABEL: test_mm256_castsi128_si256
  // CHECK: [[A:%.*]] = freeze <2 x i64> poison
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_castsi128_si256(A);
}

__m256d test_mm256_castsi256_pd(__m256i A) {
  // CHECK-LABEL: test_mm256_castsi256_pd
  return _mm256_castsi256_pd(A);
}

__m256 test_mm256_castsi256_ps(__m256i A) {
  // CHECK-LABEL: test_mm256_castsi256_ps
  return _mm256_castsi256_ps(A);
}

__m128i test_mm256_castsi256_si128(__m256i A) {
  // CHECK-LABEL: test_mm256_castsi256_si128
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <2 x i32> <i32 0, i32 1>
  return _mm256_castsi256_si128(A);
}

__m256d test_mm256_ceil_pd(__m256d x) {
  // CHECK-LABEL: test_mm256_ceil_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 2)
  return _mm256_ceil_pd(x);
}

__m256 test_mm_ceil_ps(__m256 x) {
  // CHECK-LABEL: test_mm_ceil_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 2)
  return _mm256_ceil_ps(x);
}

__m256d test_mm256_cmp_pd_eq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_oq
  // CHECK: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}

__m256d test_mm256_cmp_pd_lt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_lt_os
  // CHECK: fcmp olt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_LT_OS);
}

__m256d test_mm256_cmp_pd_le_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_le_os
  // CHECK: fcmp ole <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_LE_OS);
}

__m256d test_mm256_cmp_pd_unord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_unord_q
  // CHECK: fcmp uno <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
}

__m256d test_mm256_cmp_pd_neq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_uq
  // CHECK: fcmp une <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ);
}

__m256d test_mm256_cmp_pd_nlt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nlt_us
  // CHECK: fcmp uge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NLT_US);
}

__m256d test_mm256_cmp_pd_nle_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nle_us
  // CHECK: fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NLE_US);
}

__m256d test_mm256_cmp_pd_ord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ord_q
  // CHECK: fcmp ord <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_ORD_Q);
}

__m256d test_mm256_cmp_pd_eq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_uq
  // CHECK: fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_EQ_UQ);
}

__m256d test_mm256_cmp_pd_nge_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nge_us
  // CHECK: fcmp ult <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NGE_US);
}

__m256d test_mm256_cmp_pd_ngt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ngt_us
  // CHECK: fcmp ule <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NGT_US);
}

__m256d test_mm256_cmp_pd_false_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_false_oq
  // CHECK: fcmp false <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_FALSE_OQ);
}

__m256d test_mm256_cmp_pd_neq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_oq
  // CHECK: fcmp one <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
}

__m256d test_mm256_cmp_pd_ge_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ge_os
  // CHECK: fcmp oge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_GE_OS);
}

__m256d test_mm256_cmp_pd_gt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_gt_os
  // CHECK: fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_GT_OS);
}

__m256d test_mm256_cmp_pd_true_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_true_uq
  // CHECK: fcmp true <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_TRUE_UQ);
}

__m256d test_mm256_cmp_pd_eq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_os
  // CHECK: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_EQ_OS);
}

__m256d test_mm256_cmp_pd_lt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_lt_oq
  // CHECK: fcmp olt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

__m256d test_mm256_cmp_pd_le_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_le_oq
  // CHECK: fcmp ole <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
}

__m256d test_mm256_cmp_pd_unord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_unord_s
  // CHECK: fcmp uno <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_UNORD_S);
}

__m256d test_mm256_cmp_pd_neq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_us
  // CHECK: fcmp une <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NEQ_US);
}

__m256d test_mm256_cmp_pd_nlt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nlt_uq
  // CHECK: fcmp uge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NLT_UQ);
}

__m256d test_mm256_cmp_pd_nle_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nle_uq
  // CHECK: fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NLE_UQ);
}

__m256d test_mm256_cmp_pd_ord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ord_s
  // CHECK: fcmp ord <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_ORD_S);
}

__m256d test_mm256_cmp_pd_eq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_eq_us
  // CHECK: fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_EQ_US);
}

__m256d test_mm256_cmp_pd_nge_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_nge_uq
  // CHECK: fcmp ult <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NGE_UQ);
}

__m256d test_mm256_cmp_pd_ngt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ngt_uq
  // CHECK: fcmp ule <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NGT_UQ);
}

__m256d test_mm256_cmp_pd_false_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_false_os
  // CHECK: fcmp false <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_FALSE_OS);
}

__m256d test_mm256_cmp_pd_neq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_neq_os
  // CHECK: fcmp one <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OS);
}

__m256d test_mm256_cmp_pd_ge_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_ge_oq
  // CHECK: fcmp oge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}

__m256d test_mm256_cmp_pd_gt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_gt_oq
  // CHECK: fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
}

__m256d test_mm256_cmp_pd_true_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_true_us
  // CHECK: fcmp true <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd(a, b, _CMP_TRUE_US);
}

__m256 test_mm256_cmp_ps_eq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_oq
  // CHECK: fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

__m256 test_mm256_cmp_ps_lt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_lt_os
  // CHECK: fcmp olt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_LT_OS);
}

__m256 test_mm256_cmp_ps_le_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_le_os
  // CHECK: fcmp ole <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

__m256 test_mm256_cmp_ps_unord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_unord_q
  // CHECK: fcmp uno <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
}

__m256 test_mm256_cmp_ps_neq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_uq
  // CHECK: fcmp une <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
}

__m256 test_mm256_cmp_ps_nlt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nlt_us
  // CHECK: fcmp uge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NLT_US);
}

__m256 test_mm256_cmp_ps_nle_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nle_us
  // CHECK: fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NLE_US);
}

__m256 test_mm256_cmp_ps_ord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ord_q
  // CHECK: fcmp ord <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_ORD_Q);
}

__m256 test_mm256_cmp_ps_eq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_uq
  // CHECK: fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_EQ_UQ);
}

__m256 test_mm256_cmp_ps_nge_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nge_us
  // CHECK: fcmp ult <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NGE_US);
}

__m256 test_mm256_cmp_ps_ngt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ngt_us
  // CHECK: fcmp ule <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NGT_US);
}

__m256 test_mm256_cmp_ps_false_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_false_oq
  // CHECK: fcmp false <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_FALSE_OQ);
}

__m256 test_mm256_cmp_ps_neq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_oq
  // CHECK: fcmp one <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}

__m256 test_mm256_cmp_ps_ge_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ge_os
  // CHECK: fcmp oge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_GE_OS);
}

__m256 test_mm256_cmp_ps_gt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_gt_os
  // CHECK: fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_GT_OS);
}

__m256 test_mm256_cmp_ps_true_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_true_uq
  // CHECK: fcmp true <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_TRUE_UQ);
}

__m256 test_mm256_cmp_ps_eq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_os
  // CHECK: fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_EQ_OS);
}

__m256 test_mm256_cmp_ps_lt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_lt_oq
  // CHECK: fcmp olt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

__m256 test_mm256_cmp_ps_le_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_le_oq
  // CHECK: fcmp ole <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

__m256 test_mm256_cmp_ps_unord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_unord_s
  // CHECK: fcmp uno <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_UNORD_S);
}

__m256 test_mm256_cmp_ps_neq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_us
  // CHECK: fcmp une <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NEQ_US);
}

__m256 test_mm256_cmp_ps_nlt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nlt_uq
  // CHECK: fcmp uge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NLT_UQ);
}

__m256 test_mm256_cmp_ps_nle_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nle_uq
  // CHECK: fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NLE_UQ);
}

__m256 test_mm256_cmp_ps_ord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ord_s
  // CHECK: fcmp ord <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_ORD_S);
}

__m256 test_mm256_cmp_ps_eq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_eq_us
  // CHECK: fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_EQ_US);
}

__m256 test_mm256_cmp_ps_nge_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_nge_uq
  // CHECK: fcmp ult <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NGE_UQ);
}

__m256 test_mm256_cmp_ps_ngt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ngt_uq
  // CHECK: fcmp ule <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NGT_UQ);
}

__m256 test_mm256_cmp_ps_false_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_false_os
  // CHECK: fcmp false <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_FALSE_OS);
}

__m256 test_mm256_cmp_ps_neq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_neq_os
  // CHECK: fcmp one <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OS);
}

__m256 test_mm256_cmp_ps_ge_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_ge_oq
  // CHECK: fcmp oge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

__m256 test_mm256_cmp_ps_gt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_gt_oq
  // CHECK: fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

__m256 test_mm256_cmp_ps_true_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_true_us
  // CHECK: fcmp true <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps(a, b, _CMP_TRUE_US);
}

__m128d test_mm_cmp_pd_eq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_uq
  // CHECK: fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_EQ_UQ);
}

__m128d test_mm_cmp_pd_nge_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nge_us
  // CHECK: fcmp ult <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NGE_US);
}

__m128d test_mm_cmp_pd_ngt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ngt_us
  // CHECK: fcmp ule <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NGT_US);
}

__m128d test_mm_cmp_pd_false_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_false_oq
  // CHECK: fcmp false <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_FALSE_OQ);
}

__m128d test_mm_cmp_pd_neq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_oq
  // CHECK: fcmp one <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NEQ_OQ);
}

__m128d test_mm_cmp_pd_ge_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ge_os
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_GE_OS);
}

__m128d test_mm_cmp_pd_gt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_gt_os
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_GT_OS);
}

__m128d test_mm_cmp_pd_true_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_true_uq
  // CHECK: fcmp true <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_TRUE_UQ);
}

__m128d test_mm_cmp_pd_eq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_os
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_EQ_OS);
}

__m128d test_mm_cmp_pd_lt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_lt_oq
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_LT_OQ);
}

__m128d test_mm_cmp_pd_le_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_le_oq
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_LE_OQ);
}

__m128d test_mm_cmp_pd_unord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_unord_s
  // CHECK: fcmp uno <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_UNORD_S);
}

__m128d test_mm_cmp_pd_neq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_us
  // CHECK: fcmp une <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NEQ_US);
}

__m128d test_mm_cmp_pd_nlt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nlt_uq
  // CHECK: fcmp uge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NLT_UQ);
}

__m128d test_mm_cmp_pd_nle_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nle_uq
  // CHECK: fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NLE_UQ);
}

__m128d test_mm_cmp_pd_ord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ord_s
  // CHECK: fcmp ord <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_ORD_S);
}

__m128d test_mm_cmp_pd_eq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_us
  // CHECK: fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_EQ_US);
}

__m128d test_mm_cmp_pd_nge_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nge_uq
  // CHECK: fcmp ult <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NGE_UQ);
}

__m128d test_mm_cmp_pd_ngt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ngt_uq
  // CHECK: fcmp ule <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NGT_UQ);
}

__m128d test_mm_cmp_pd_false_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_false_os
  // CHECK: fcmp false <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_FALSE_OS);
}

__m128d test_mm_cmp_pd_neq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_os
  // CHECK: fcmp one <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NEQ_OS);
}

__m128d test_mm_cmp_pd_ge_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ge_oq
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_GE_OQ);
}

__m128d test_mm_cmp_pd_gt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_gt_oq
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_GT_OQ);
}

__m128d test_mm_cmp_pd_true_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_true_us
  // CHECK: fcmp true <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_TRUE_US);
}

__m128 test_mm_cmp_ps_eq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_uq
  // CHECK: fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_EQ_UQ);
}

__m128 test_mm_cmp_ps_nge_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nge_us
  // CHECK: fcmp ult <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NGE_US);
}

__m128 test_mm_cmp_ps_ngt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ngt_us
  // CHECK: fcmp ule <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NGT_US);
}

__m128 test_mm_cmp_ps_false_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_false_oq
  // CHECK: fcmp false <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_FALSE_OQ);
}

__m128 test_mm_cmp_ps_neq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_oq
  // CHECK: fcmp one <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NEQ_OQ);
}

__m128 test_mm_cmp_ps_ge_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ge_os
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GE_OS);
}

__m128 test_mm_cmp_ps_gt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_gt_os
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GT_OS);
}

__m128 test_mm_cmp_ps_true_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_true_uq
  // CHECK: fcmp true <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_TRUE_UQ);
}

__m128 test_mm_cmp_ps_eq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_os
  // CHECK: fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_EQ_OS);
}

__m128 test_mm_cmp_ps_lt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_lt_oq
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_LT_OQ);
}

__m128 test_mm_cmp_ps_le_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_le_oq
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_LE_OQ);
}

__m128 test_mm_cmp_ps_unord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_unord_s
  // CHECK: fcmp uno <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_UNORD_S);
}

__m128 test_mm_cmp_ps_neq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_us
  // CHECK: fcmp une <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NEQ_US);
}

__m128 test_mm_cmp_ps_nlt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nlt_uq
  // CHECK: fcmp uge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NLT_UQ);
}

__m128 test_mm_cmp_ps_nle_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nle_uq
  // CHECK: fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NLE_UQ);
}

__m128 test_mm_cmp_ps_ord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ord_s
  // CHECK: fcmp ord <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_ORD_S);
}

__m128 test_mm_cmp_ps_eq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_eq_us
  // CHECK: fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_EQ_US);
}

__m128 test_mm_cmp_ps_nge_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_nge_uq
  // CHECK: fcmp ult <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NGE_UQ);
}

__m128 test_mm_cmp_ps_ngt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ngt_uq
  // CHECK: fcmp ule <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NGT_UQ);
}

__m128 test_mm_cmp_ps_false_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_false_os
  // CHECK: fcmp false <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_FALSE_OS);
}

__m128 test_mm_cmp_ps_neq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_neq_os
  // CHECK: fcmp one <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_NEQ_OS);
}

__m128 test_mm_cmp_ps_ge_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_ge_oq
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GE_OQ);
}

__m128 test_mm_cmp_ps_gt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_gt_oq
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_GT_OQ);
}

__m128 test_mm_cmp_ps_true_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_true_us
  // CHECK: fcmp true <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps(a, b, _CMP_TRUE_US);
}

__m128d test_mm_cmp_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmp_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 13)
  return _mm_cmp_sd(A, B, _CMP_GE_OS);
}

__m128 test_mm_cmp_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_cmp_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 13)
  return _mm_cmp_ss(A, B, _CMP_GE_OS);
}

__m256d test_mm256_cvtepi32_pd(__m128i A) {
  // CHECK-LABEL: test_mm256_cvtepi32_pd
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x double>
  return _mm256_cvtepi32_pd(A);
}

__m256 test_mm256_cvtepi32_ps(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepi32_ps
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x float>
  return _mm256_cvtepi32_ps(A);
}

__m128i test_mm256_cvtpd_epi32(__m256d A) {
  // CHECK-LABEL: test_mm256_cvtpd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double> %{{.*}})
  return _mm256_cvtpd_epi32(A);
}

__m128 test_mm256_cvtpd_ps(__m256d A) {
  // CHECK-LABEL: test_mm256_cvtpd_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx.cvt.pd2.ps.256(<4 x double> %{{.*}})
  return _mm256_cvtpd_ps(A);
}

__m256i test_mm256_cvtps_epi32(__m256 A) {
  // CHECK-LABEL: test_mm256_cvtps_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float> %{{.*}})
  return _mm256_cvtps_epi32(A);
}

__m256d test_mm256_cvtps_pd(__m128 A) {
  // CHECK-LABEL: test_mm256_cvtps_pd
  // CHECK: fpext <4 x float> %{{.*}} to <4 x double>
  return _mm256_cvtps_pd(A);
}

double test_mm256_cvtsd_f64(__m256d __a) {
  // CHECK-LABEL: test_mm256_cvtsd_f64
  // CHECK: extractelement <4 x double> %{{.*}}, i32 0
  return _mm256_cvtsd_f64(__a);
}

int test_mm256_cvtsi256_si32(__m256i __a) {
  // CHECK-LABEL: test_mm256_cvtsi256_si32
  // CHECK: extractelement <8 x i32> %{{.*}}, i32 0
  return _mm256_cvtsi256_si32(__a);
}

float test_mm256_cvtss_f32(__m256 __a) {
  // CHECK-LABEL: test_mm256_cvtss_f32
  // CHECK: extractelement <8 x float> %{{.*}}, i32 0
  return _mm256_cvtss_f32(__a);
}

__m128i test_mm256_cvttpd_epi32(__m256d A) {
  // CHECK-LABEL: test_mm256_cvttpd_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx.cvtt.pd2dq.256(<4 x double> %{{.*}})
  return _mm256_cvttpd_epi32(A);
}

__m256i test_mm256_cvttps_epi32(__m256 A) {
  // CHECK-LABEL: test_mm256_cvttps_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx.cvtt.ps2dq.256(<8 x float> %{{.*}})
  return _mm256_cvttps_epi32(A);
}

__m256d test_mm256_div_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_div_pd
  // CHECK: fdiv <4 x double>
  return _mm256_div_pd(A, B);
}

__m256 test_mm256_div_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_div_ps
  // CHECK: fdiv <8 x float>
  return _mm256_div_ps(A, B);
}

__m256 test_mm256_dp_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_dp_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.dp.ps.256(<8 x float> {{.*}}, <8 x float> {{.*}}, i8 7)
  return _mm256_dp_ps(A, B, 7);
}

int test_mm256_extract_epi8(__m256i A) {
  // CHECK-LABEL: test_mm256_extract_epi8
  // CHECK: extractelement <32 x i8> %{{.*}}, {{i32|i64}} 31
  // CHECK: zext i8 %{{.*}} to i32
  return _mm256_extract_epi8(A, 31);
}

int test_mm256_extract_epi16(__m256i A) {
  // CHECK-LABEL: test_mm256_extract_epi16
  // CHECK: extractelement <16 x i16> %{{.*}}, {{i32|i64}} 15
  // CHECK: zext i16 %{{.*}} to i32
  return _mm256_extract_epi16(A, 15);
}

int test_mm256_extract_epi32(__m256i A) {
  // CHECK-LABEL: test_mm256_extract_epi32
  // CHECK: extractelement <8 x i32> %{{.*}}, {{i32|i64}} 7
  return _mm256_extract_epi32(A, 7);
}

#if __x86_64__
long long test_mm256_extract_epi64(__m256i A) {
  // X64-LABEL: test_mm256_extract_epi64
  // X64: extractelement <4 x i64> %{{.*}}, {{i32|i64}} 3
  return _mm256_extract_epi64(A, 3);
}
#endif

__m128d test_mm256_extractf128_pd(__m256d A) {
  // CHECK-LABEL: test_mm256_extractf128_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <2 x i32> <i32 2, i32 3>
  return _mm256_extractf128_pd(A, 1);
}

__m128 test_mm256_extractf128_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_extractf128_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm256_extractf128_ps(A, 1);
}

__m128i test_mm256_extractf128_si256(__m256i A) {
  // CHECK-LABEL: test_mm256_extractf128_si256
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm256_extractf128_si256(A, 1);
}

__m256d test_mm256_floor_pd(__m256d x) {
  // CHECK-LABEL: test_mm256_floor_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 1)
  return _mm256_floor_pd(x);
}

__m256 test_mm_floor_ps(__m256 x) {
  // CHECK-LABEL: test_mm_floor_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 1)
  return _mm256_floor_ps(x);
}

__m256d test_mm256_hadd_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_hadd_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_hadd_pd(A, B);
}

__m256 test_mm256_hadd_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_hadd_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_hadd_ps(A, B);
}

__m256d test_mm256_hsub_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_hsub_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_hsub_pd(A, B);
}

__m256 test_mm256_hsub_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_hsub_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_hsub_ps(A, B);
}

__m256i test_mm256_insert_epi8(__m256i x, char b) {
  // CHECK-LABEL: test_mm256_insert_epi8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 14
  return _mm256_insert_epi8(x, b, 14);
}

__m256i test_mm256_insert_epi16(__m256i x, int b) {
  // CHECK-LABEL: test_mm256_insert_epi16
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, {{i32|i64}} 4
  return _mm256_insert_epi16(x, b, 4);
}

__m256i test_mm256_insert_epi32(__m256i x, int b) {
  // CHECK-LABEL: test_mm256_insert_epi32
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 5
  return _mm256_insert_epi32(x, b, 5);
}

#if __x86_64__
__m256i test_mm256_insert_epi64(__m256i x, long long b) {
  // X64-LABEL: test_mm256_insert_epi64
  // X64: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 2
  return _mm256_insert_epi64(x, b, 2);
}
#endif

__m256d test_mm256_insertf128_pd(__m256d A, __m128d B) {
  // CHECK-LABEL: test_mm256_insertf128_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_insertf128_pd(A, B, 0);
}

__m256 test_mm256_insertf128_ps(__m256 A, __m128 B) {
  // CHECK-LABEL: test_mm256_insertf128_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf128_ps(A, B, 1);
}

__m256i test_mm256_insertf128_si256(__m256i A, __m128i B) {
  // CHECK-LABEL: test_mm256_insertf128_si256
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  return _mm256_insertf128_si256(A, B, 0);
}

__m256i test_mm256_lddqu_si256(__m256i* A) {
  // CHECK-LABEL: test_mm256_lddqu_si256
  // CHECK: call <32 x i8> @llvm.x86.avx.ldu.dq.256(ptr %{{.*}})
  return _mm256_lddqu_si256(A);
}

__m256d test_mm256_load_pd(double* A) {
  // CHECK-LABEL: test_mm256_load_pd
  // CHECK: load <4 x double>, ptr %{{.*}}, align 32
  return _mm256_load_pd(A);
}

__m256 test_mm256_load_ps(float* A) {
  // CHECK-LABEL: test_mm256_load_ps
  // CHECK: load <8 x float>, ptr %{{.*}}, align 32
  return _mm256_load_ps(A);
}

__m256i test_mm256_load_si256(__m256i* A) {
  // CHECK-LABEL: test_mm256_load_si256
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 32
  return _mm256_load_si256(A);
}

__m256d test_mm256_loadu_pd(double* A) {
  // CHECK-LABEL: test_mm256_loadu_pd
  // CHECK: load <4 x double>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_pd(A);
}

__m256 test_mm256_loadu_ps(float* A) {
  // CHECK-LABEL: test_mm256_loadu_ps
  // CHECK: load <8 x float>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_ps(A);
}

__m256i test_mm256_loadu_si256(__m256i* A) {
  // CHECK-LABEL: test_mm256_loadu_si256
  // CHECK: load <4 x i64>, ptr %{{.+}}, align 1{{$}}
  return _mm256_loadu_si256(A);
}

__m256 test_mm256_loadu2_m128(float* A, float* B) {
  // CHECK-LABEL: test_mm256_loadu2_m128
  // CHECK: load <4 x float>, ptr %{{.*}}, align 1{{$}}
  // CHECK: load <4 x float>, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_loadu2_m128(A, B);
}

__m256d test_mm256_loadu2_m128d(double* A, double* B) {
  // CHECK-LABEL: test_mm256_loadu2_m128d
  // CHECK: load <2 x double>, ptr %{{.*}}, align 1{{$}}
  // CHECK: load <2 x double>, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_loadu2_m128d(A, B);
}

__m256i test_mm256_loadu2_m128i(__m128i* A, __m128i* B) {
  // CHECK-LABEL: test_mm256_loadu2_m128i
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_loadu2_m128i(A, B);
}

__m128d test_mm_maskload_pd(double* A, __m128i B) {
  // CHECK-LABEL: test_mm_maskload_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.avx.maskload.pd(ptr %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskload_pd(A, B);
}

__m256d test_mm256_maskload_pd(double* A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskload_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.maskload.pd.256(ptr %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskload_pd(A, B);
}

__m128 test_mm_maskload_ps(float* A, __m128i B) {
  // CHECK-LABEL: test_mm_maskload_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx.maskload.ps(ptr %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskload_ps(A, B);
}

__m256 test_mm256_maskload_ps(float* A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskload_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.maskload.ps.256(ptr %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskload_ps(A, B);
}

void test_mm_maskstore_pd(double* A, __m128i B, __m128d C) {
  // CHECK-LABEL: test_mm_maskstore_pd
  // CHECK: call void @llvm.x86.avx.maskstore.pd(ptr %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}})
  _mm_maskstore_pd(A, B, C);
}

void test_mm256_maskstore_pd(double* A, __m256i B, __m256d C) {
  // CHECK-LABEL: test_mm256_maskstore_pd
  // CHECK: call void @llvm.x86.avx.maskstore.pd.256(ptr %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}})
  _mm256_maskstore_pd(A, B, C);
}

void test_mm_maskstore_ps(float* A, __m128i B, __m128 C) {
  // CHECK-LABEL: test_mm_maskstore_ps
  // CHECK: call void @llvm.x86.avx.maskstore.ps(ptr %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}})
  _mm_maskstore_ps(A, B, C);
}

void test_mm256_maskstore_ps(float* A, __m256i B, __m256 C) {
  // CHECK-LABEL: test_mm256_maskstore_ps
  // CHECK: call void @llvm.x86.avx.maskstore.ps.256(ptr %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}})
  _mm256_maskstore_ps(A, B, C);
}

__m256d test_mm256_max_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_max_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_max_pd(A, B);
}

__m256 test_mm256_max_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_max_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_max_ps(A, B);
}

__m256d test_mm256_min_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_min_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_min_pd(A, B);
}

__m256 test_mm256_min_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_min_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_min_ps(A, B);
}

__m256d test_mm256_movedup_pd(__m256d A) {
  // CHECK-LABEL: test_mm256_movedup_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  return _mm256_movedup_pd(A);
}

__m256 test_mm256_movehdup_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_movehdup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  return _mm256_movehdup_ps(A);
}

__m256 test_mm256_moveldup_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_moveldup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  return _mm256_moveldup_ps(A);
}

int test_mm256_movemask_pd(__m256d A) {
  // CHECK-LABEL: test_mm256_movemask_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %{{.*}})
  return _mm256_movemask_pd(A);
}

int test_mm256_movemask_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_movemask_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %{{.*}})
  return _mm256_movemask_ps(A);
}

__m256d test_mm256_mul_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_mul_pd
  // CHECK: fmul <4 x double>
  return _mm256_mul_pd(A, B);
}

__m256 test_mm256_mul_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_mul_ps
  // CHECK: fmul <8 x float>
  return _mm256_mul_ps(A, B);
}

__m256d test_mm256_or_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_or_pd
  // CHECK: or <4 x i64>
  return _mm256_or_pd(A, B);
}

__m256 test_mm256_or_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_or_ps
  // CHECK: or <8 x i32>
  return _mm256_or_ps(A, B);
}

__m128d test_mm_permute_pd(__m128d A) {
  // CHECK-LABEL: test_mm_permute_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  return _mm_permute_pd(A, 1);
}

__m256d test_mm256_permute_pd(__m256d A) {
  // CHECK-LABEL: test_mm256_permute_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  return _mm256_permute_pd(A, 5);
}

__m128 test_mm_permute_ps(__m128 A) {
  // CHECK-LABEL: test_mm_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  return _mm_permute_ps(A, 0x1b);
}

// Test case for PR12401
__m128 test2_mm_permute_ps(__m128 a) {
  // CHECK-LABEL: test2_mm_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 2, i32 1, i32 2, i32 3>
  return _mm_permute_ps(a, 0xe6);
}

__m256 test_mm256_permute_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_permute_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  return _mm256_permute_ps(A, 0x1b);
}

__m256d test_mm256_permute2f128_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_permute2f128_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  return _mm256_permute2f128_pd(A, B, 0x31);
}

__m256 test_mm256_permute2f128_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_permute2f128_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_permute2f128_ps(A, B, 0x13);
}

__m256i test_mm256_permute2f128_si256(__m256i A, __m256i B) {
  // CHECK-LABEL: test_mm256_permute2f128_si256
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_permute2f128_si256(A, B, 0x20);
}

__m128d test_mm_permutevar_pd(__m128d A, __m128i B) {
  // CHECK-LABEL: test_mm_permutevar_pd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_permutevar_pd(A, B);
}

__m256d test_mm256_permutevar_pd(__m256d A, __m256i B) {
  // CHECK-LABEL: test_mm256_permutevar_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_permutevar_pd(A, B);
}

__m128 test_mm_permutevar_ps(__m128 A, __m128i B) {
  // CHECK-LABEL: test_mm_permutevar_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_permutevar_ps(A, B);
}

__m256 test_mm256_permutevar_ps(__m256 A, __m256i B) {
  // CHECK-LABEL: test_mm256_permutevar_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_permutevar_ps(A, B);
}

__m256 test_mm256_rcp_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_rcp_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %{{.*}})
  return _mm256_rcp_ps(A);
}

__m256d test_mm256_round_pd(__m256d x) {
  // CHECK-LABEL: test_mm256_round_pd
  // CHECK: call {{.*}}<4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 4)
  return _mm256_round_pd(x, 4);
}

__m256 test_mm256_round_ps(__m256 x) {
  // CHECK-LABEL: test_mm256_round_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 4)
  return _mm256_round_ps(x, 4);
}

__m256 test_mm256_rsqrt_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_rsqrt_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float> %{{.*}})
  return _mm256_rsqrt_ps(A);
}

__m256i test_mm256_set_epi8(char A0, char A1, char A2, char A3, char A4, char A5, char A6, char A7,
                            char A8, char A9, char A10, char A11, char A12, char A13, char A14, char A15,
                            char A16, char A17, char A18, char A19, char A20, char A21, char A22, char A23,
                            char A24, char A25, char A26, char A27, char A28, char A29, char A30, char A31) {
  // CHECK-LABEL: test_mm256_set_epi8
  // CHECK: insertelement <32 x i8> poison, i8 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  return _mm256_set_epi8(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31);
}

__m256i test_mm256_set_epi16(short A0, short A1, short A2, short A3, short A4, short A5, short A6, short A7,
                             short A8, short A9, short A10, short A11, short A12, short A13, short A14, short A15) {
  // CHECK-LABEL: test_mm256_set_epi16
  // CHECK: insertelement <16 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  return _mm256_set_epi16(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
}

__m256i test_mm256_set_epi32(int A0, int A1, int A2, int A3, int A4, int A5, int A6, int A7) {
  // CHECK-LABEL: test_mm256_set_epi32
  // CHECK: insertelement <8 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  return _mm256_set_epi32(A0, A1, A2, A3, A4, A5, A6, A7);
}

__m256i test_mm256_set_epi64x(long long A0, long long A1, long long A2, long long A3) {
  // CHECK-LABEL: test_mm256_set_epi64x
  // CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  return _mm256_set_epi64x(A0, A1, A2, A3);
}

__m256 test_mm256_set_m128(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm256_set_m128
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_set_m128(A, B);
}

__m256d test_mm256_set_m128d(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm256_set_m128d
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_set_m128d(A, B);
}

__m256i test_mm256_set_m128i(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm256_set_m128i
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_set_m128i(A, B);
}

__m256d test_mm256_set_pd(double A0, double A1, double A2, double A3) {
  // CHECK-LABEL: test_mm256_set_pd
  // CHECK: insertelement <4 x double> poison, double %{{.*}}, i32 0
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 1
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 2
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 3
  return _mm256_set_pd(A0, A1, A2, A3);
}
TEST_CONSTEXPR(match_m256d(_mm256_set_pd(-100.0, +90.0, -50.0, +1.0), +1.0, -50.0, +90.0, -100.0));

__m256 test_mm256_set_ps(float A0, float A1, float A2, float A3, float A4, float A5, float A6, float A7) {
  // CHECK-LABEL: test_mm256_set_ps
  // CHECK: insertelement <8 x float> poison, float %{{.*}}, i32 0
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 3
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 4
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 5
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 6
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 7
  return _mm256_set_ps(A0, A1, A2, A3, A4, A5, A6, A7);
}
TEST_CONSTEXPR(match_m256(_mm256_set_ps(-1.0f, +2.0f, -3.0f, +4.0f, -5.0f, +6.0f, -7.0f, +8.0f), +8.0f, -7.0f, +6.0f, -5.0f, +4.0f, -3.0f, +2.0f, -1.0f));

__m256i test_mm256_set1_epi8(char A) {
  // CHECK-LABEL: test_mm256_set1_epi8
  // CHECK: insertelement <32 x i8> poison, i8 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  return _mm256_set1_epi8(A);
}

__m256i test_mm256_set1_epi16(short A) {
  // CHECK-LABEL: test_mm256_set1_epi16
  // CHECK: insertelement <16 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  return _mm256_set1_epi16(A);
}

__m256i test_mm256_set1_epi32(int A) {
  // CHECK-LABEL: test_mm256_set1_epi32
  // CHECK: insertelement <8 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  return _mm256_set1_epi32(A);
}

__m256i test_mm256_set1_epi64x(long long A) {
  // CHECK-LABEL: test_mm256_set1_epi64x
  // CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  return _mm256_set1_epi64x(A);
}

__m256d test_mm256_set1_pd(double A) {
  // CHECK-LABEL: test_mm256_set1_pd
  // CHECK: insertelement <4 x double> poison, double %{{.*}}, i32 0
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 1
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 2
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 3
  return _mm256_set1_pd(A);
}
TEST_CONSTEXPR(match_m256d(_mm256_set1_pd(+42.0), +42.0, +42.0, +42.0, +42.0));

__m256 test_mm256_set1_ps(float A) {
  // CHECK-LABEL: test_mm256_set1_ps
  // CHECK: insertelement <8 x float> poison, float %{{.*}}, i32 0
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 3
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 4
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 5
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 6
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 7
  return _mm256_set1_ps(A);
}
TEST_CONSTEXPR(match_m256(_mm256_set1_ps(-101.0f), -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f));

__m256i test_mm256_setr_epi8(char A0, char A1, char A2, char A3, char A4, char A5, char A6, char A7,
                             char A8, char A9, char A10, char A11, char A12, char A13, char A14, char A15,
                             char A16, char A17, char A18, char A19, char A20, char A21, char A22, char A23,
                             char A24, char A25, char A26, char A27, char A28, char A29, char A30, char A31) {
  // CHECK-LABEL: test_mm256_setr_epi8
  // CHECK: insertelement <32 x i8> poison, i8 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  return _mm256_setr_epi8(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31);
}

__m256i test_mm256_setr_epi16(short A0, short A1, short A2, short A3, short A4, short A5, short A6, short A7,
                              short A8, short A9, short A10, short A11, short A12, short A13, short A14, short A15) {
  // CHECK-LABEL: test_mm256_setr_epi16
  // CHECK: insertelement <16 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  return _mm256_setr_epi16(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
}

__m256i test_mm256_setr_epi32(int A0, int A1, int A2, int A3, int A4, int A5, int A6, int A7) {
  // CHECK-LABEL: test_mm256_setr_epi32
  // CHECK: insertelement <8 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  return _mm256_setr_epi32(A0, A1, A2, A3, A4, A5, A6, A7);
}

__m256i test_mm256_setr_epi64x(long long A0, long long A1, long long A2, long long A3) {
  // CHECK-LABEL: test_mm256_setr_epi64x
  // CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  return _mm256_setr_epi64x(A0, A1, A2, A3);
}

__m256 test_mm256_setr_m128(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm256_setr_m128
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_setr_m128(A, B);
}

__m256d test_mm256_setr_m128d(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm256_setr_m128d
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_setr_m128d(A, B);
}

__m256i test_mm256_setr_m128i(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm256_setr_m128i
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_setr_m128i(A, B);
}

__m256d test_mm256_setr_pd(double A0, double A1, double A2, double A3) {
  // CHECK-LABEL: test_mm256_setr_pd
  // CHECK: insertelement <4 x double> poison, double %{{.*}}, i32 0
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 1
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 2
  // CHECK: insertelement <4 x double> %{{.*}}, double %{{.*}}, i32 3
  return _mm256_setr_pd(A0, A1, A2, A3);
}
TEST_CONSTEXPR(match_m256d(_mm256_setr_pd(-100.0, +90.0, -50.0, +1.0), -100.0, +90.0, -50.0, +1.0));

__m256 test_mm256_setr_ps(float A0, float A1, float A2, float A3, float A4, float A5, float A6, float A7) {
  // CHECK-LABEL: test_mm256_setr_ps
  // CHECK: insertelement <8 x float> poison, float %{{.*}}, i32 0
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 3
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 4
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 5
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 6
  // CHECK: insertelement <8 x float> %{{.*}}, float %{{.*}}, i32 7
  return _mm256_setr_ps(A0, A1, A2, A3, A4, A5, A6, A7);
}
TEST_CONSTEXPR(match_m256(_mm256_setr_ps(-1.0f, +2.0f, -3.0f, +4.0f, -5.0f, +6.0f, -7.0f, +8.0f), -1.0f, +2.0f, -3.0f, +4.0f, -5.0f, +6.0f, -7.0f, +8.0f));

__m256d test_mm256_setzero_pd(void) {
  // CHECK-LABEL: test_mm256_setzero_pd
  // CHECK: store <4 x double> zeroinitializer
  return _mm256_setzero_pd();
}
TEST_CONSTEXPR(match_m256d(_mm256_setzero_pd(), +0.0, +0.0, +0.0, +0.0));

__m256 test_mm256_setzero_ps(void) {
  // CHECK-LABEL: test_mm256_setzero_ps
  // CHECK: store <8 x float> zeroinitializer
  return _mm256_setzero_ps();
}
TEST_CONSTEXPR(match_m256(_mm256_setzero_ps(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m256i test_mm256_setzero_si256(void) {
  // CHECK-LABEL: test_mm256_setzero_si256
  // CHECK: store <4 x i64> zeroinitializer
  return _mm256_setzero_si256();
}
TEST_CONSTEXPR(match_m256i(_mm256_setzero_si256(), 0, 0, 0, 0));

__m256d test_mm256_shuffle_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_shuffle_pd(A, B, 0);
}

__m256 test_mm256_shuffle_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 8, i32 8, i32 4, i32 4, i32 12, i32 12>
  return _mm256_shuffle_ps(A, B, 0);
}

__m256d test_mm256_sqrt_pd(__m256d A) {
  // CHECK-LABEL: test_mm256_sqrt_pd
  // CHECK: call {{.*}}<4 x double> @llvm.sqrt.v4f64(<4 x double> %{{.*}})
  return _mm256_sqrt_pd(A);
}

__m256 test_mm256_sqrt_ps(__m256 A) {
  // CHECK-LABEL: test_mm256_sqrt_ps
  // CHECK: call {{.*}}<8 x float> @llvm.sqrt.v8f32(<8 x float> %{{.*}})
  return _mm256_sqrt_ps(A);
}

void test_mm256_store_pd(double* A, __m256d B) {
  // CHECK-LABEL: test_mm256_store_pd
  // CHECK: store <4 x double> %{{.*}}, ptr %{{.*}}, align 32
  _mm256_store_pd(A, B);
}

void test_mm256_store_ps(float* A, __m256 B) {
  // CHECK-LABEL: test_mm256_store_ps
  // CHECK: store <8 x float> %{{.*}}, ptr %{{.*}}, align 32
  _mm256_store_ps(A, B);
}

void test_mm256_store_si256(__m256i* A, __m256i B) {
  // CHECK-LABEL: test_mm256_store_si256
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 32
  _mm256_store_si256(A, B);
}

void test_mm256_storeu_pd(double* A, __m256d B) {
  // CHECK-LABEL: test_mm256_storeu_pd
  // CHECK:   store <4 x double> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_pd(A, B);
}

void test_mm256_storeu_ps(float* A, __m256 B) {
  // CHECK-LABEL: test_mm256_storeu_ps
  // CHECK: store <8 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_ps(A, B);
}

void test_mm256_storeu_si256(__m256i* A, __m256i B) {
  // CHECK-LABEL: test_mm256_storeu_si256
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_si256(A, B);
}

void test_mm256_storeu2_m128(float* A, float* B, __m256 C) {
  // CHECK-LABEL: test_mm256_storeu2_m128
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm256_storeu2_m128(A, B, C);
}

void test_mm256_storeu2_m128d(double* A, double* B, __m256d C) {
  // CHECK-LABEL: test_mm256_storeu2_m128d
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <2 x i32> <i32 2, i32 3>
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm256_storeu2_m128d(A, B, C);
}

void test_mm256_storeu2_m128i(__m128i* A, __m128i* B, __m256i C) {
  // CHECK-LABEL: test_mm256_storeu2_m128i
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm256_storeu2_m128i(A, B, C);
}

void test_mm256_stream_pd(double* A, __m256d B) {
  // CHECK-LABEL: test_mm256_stream_pd
  // CHECK: store <4 x double> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_pd(A, B);
}

void test_mm256_stream_pd_void(void *A, __m256d B) {
  // CHECK-LABEL: test_mm256_stream_pd_void
  // CHECK: store <4 x double> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_pd(A, B);
}

void test_mm256_stream_ps(float* A, __m256 B) {
  // CHECK-LABEL: test_mm256_stream_ps
  // CHECK: store <8 x float> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_ps(A, B);
}

void test_mm256_stream_ps_void(void *A, __m256 B) {
  // CHECK-LABEL: test_mm256_stream_ps_void
  // CHECK: store <8 x float> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_ps(A, B);
}

void test_mm256_stream_si256(__m256i* A, __m256i B) {
  // CHECK-LABEL: test_mm256_stream_si256
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_si256(A, B);
}

void test_mm256_stream_si256_void(void *A, __m256i B) {
  // CHECK-LABEL: test_mm256_stream_si256_void
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 32, !nontemporal
  _mm256_stream_si256(A, B);
}

__m256d test_mm256_sub_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_sub_pd
  // CHECK: fsub <4 x double>
  return _mm256_sub_pd(A, B);
}

__m256 test_mm256_sub_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_sub_ps
  // CHECK: fsub <8 x float>
  return _mm256_sub_ps(A, B);
}

int test_mm_testc_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_testc_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestc.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_testc_pd(A, B);
}

int test_mm256_testc_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_testc_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestc.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_testc_pd(A, B);
}

int test_mm_testc_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_testc_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestc.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_testc_ps(A, B);
}

int test_mm256_testc_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_testc_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestc.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_testc_ps(A, B);
}

int test_mm256_testc_si256(__m256i A, __m256i B) {
  // CHECK-LABEL: test_mm256_testc_si256
  // CHECK: call {{.*}}i32 @llvm.x86.avx.ptestc.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_testc_si256(A, B);
}

int test_mm_testnzc_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_testnzc_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestnzc.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_testnzc_pd(A, B);
}

int test_mm256_testnzc_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_testnzc_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestnzc.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_testnzc_pd(A, B);
}

int test_mm_testnzc_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_testnzc_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestnzc.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_testnzc_ps(A, B);
}

int test_mm256_testnzc_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_testnzc_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestnzc.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_testnzc_ps(A, B);
}

int test_mm256_testnzc_si256(__m256i A, __m256i B) {
  // CHECK-LABEL: test_mm256_testnzc_si256
  // CHECK: call {{.*}}i32 @llvm.x86.avx.ptestnzc.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_testnzc_si256(A, B);
}

int test_mm_testz_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_testz_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestz.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_testz_pd(A, B);
}

int test_mm256_testz_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_testz_pd
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestz.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_testz_pd(A, B);
}

int test_mm_testz_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_testz_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestz.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_testz_ps(A, B);
}

int test_mm256_testz_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_testz_ps
  // CHECK: call {{.*}}i32 @llvm.x86.avx.vtestz.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_testz_ps(A, B);
}

int test_mm256_testz_si256(__m256i A, __m256i B) {
  // CHECK-LABEL: test_mm256_testz_si256
  // CHECK: call {{.*}}i32 @llvm.x86.avx.ptestz.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_testz_si256(A, B);
}

__m256 test_mm256_undefined_ps(void) {
  // X64-LABEL: test_mm256_undefined_ps
  // X64: ret <8 x float> zeroinitializer
  //
  // X86-LABEL: test_mm256_undefined_ps
  // X86: store <8 x float> zeroinitializer
  return _mm256_undefined_ps();
}

__m256d test_mm256_undefined_pd(void) {
  // X64-LABEL: test_mm256_undefined_pd
  // X64: ret <4 x double> zeroinitializer
  //
  // X86-LABEL: test_mm256_undefined_pd
  // X86: store <4 x double> zeroinitializer
  return _mm256_undefined_pd();
}

__m256i test_mm256_undefined_si256(void) {
  // X64-LABEL: test_mm256_undefined_si256
  // X64: ret <4 x i64> zeroinitializer
  //
  // X86-LABEL: test_mm256_undefined_si256
  // X86: store <4 x i64> zeroinitializer
  return _mm256_undefined_si256();
}

__m256d test_mm256_unpackhi_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_unpackhi_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  return _mm256_unpackhi_pd(A, B);
}

__m256 test_mm256_unpackhi_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_unpackhi_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  return _mm256_unpackhi_ps(A, B);
}

__m256d test_mm256_unpacklo_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_unpacklo_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_unpacklo_pd(A, B);
}

__m256 test_mm256_unpacklo_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_unpacklo_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  return _mm256_unpacklo_ps(A, B);
}

__m256d test_mm256_xor_pd(__m256d A, __m256d B) {
  // CHECK-LABEL: test_mm256_xor_pd
  // CHECK: xor <4 x i64>
  return _mm256_xor_pd(A, B);
}

__m256 test_mm256_xor_ps(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_xor_ps
  // CHECK: xor <8 x i32>
  return _mm256_xor_ps(A, B);
}

void test_mm256_zeroall(void) {
  // CHECK-LABEL: test_mm256_zeroall
  // CHECK: call void @llvm.x86.avx.vzeroall()
  return _mm256_zeroall();
}

void test_mm256_zeroupper(void) {
  // CHECK-LABEL: test_mm256_zeroupper
  // CHECK: call void @llvm.x86.avx.vzeroupper()
  return _mm256_zeroupper();
}

__m256d test_mm256_zextpd128_pd256(__m128d A) {
  // CHECK-LABEL: test_mm256_zextpd128_pd256
  // CHECK: store <2 x double> zeroinitializer
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_zextpd128_pd256(A);
}

__m256 test_mm256_zextps128_ps256(__m128 A) {
  // CHECK-LABEL: test_mm256_zextps128_ps256
  // CHECK: store <4 x float> zeroinitializer
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_zextps128_ps256(A);
}

__m256i test_mm256_zextsi128_si256(__m128i A) {
  // CHECK-LABEL: test_mm256_zextsi128_si256
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm256_zextsi128_si256(A);
}

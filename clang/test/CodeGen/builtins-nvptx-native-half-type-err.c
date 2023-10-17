// REQUIRES: nvptx-registered-target
//
// RUN: not %clang_cc1 -fsyntax-only -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_86 -target-feature +ptx72 -fcuda-is-device -x cuda -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_ERROR %s

#define __device__ __attribute__((device))
typedef __fp16 __fp16v2 __attribute__((ext_vector_type(2)));

__device__ void nvvm_native_half_types(void *a, void*b, void*c, __fp16* out) {
  __fp16v2 resv2 = {0, 0};
  *out += __nvvm_ex2_approx_f16(*(__fp16 *)a);
  resv2 = __nvvm_ex2_approx_f16x2(*(__fp16v2*)a);

  *out += __nvvm_fma_rn_relu_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_ftz_relu_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16 *)c);
  resv2 += __nvvm_fma_rn_relu_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_relu_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  *out += __nvvm_fma_rn_ftz_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_sat_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_ftz_sat_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  resv2 += __nvvm_fma_rn_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_sat_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_sat_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);

  *out += __nvvm_fmin_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_nan_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_nan_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmin_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  *out += __nvvm_fmin_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmin_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);

  *out += __nvvm_fmax_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_nan_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_nan_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmax_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  *out += __nvvm_fmax_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmax_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);

  *out += __nvvm_ldg_h((__fp16 *)a);
  resv2 += __nvvm_ldg_h2((__fp16v2 *)a);

  *out += __nvvm_ldu_h((__fp16 *)a);
  resv2 += __nvvm_ldu_h2((__fp16v2 *)a);

  *out += resv2[0] + resv2[1];
}

// CHECK_ERROR: error: __nvvm_ex2_approx_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_ex2_approx_f16x2 requires native half type support.

// CHECK_ERROR: error: __nvvm_fma_rn_relu_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_relu_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_relu_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_relu_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_sat_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_sat_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_sat_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fma_rn_ftz_sat_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_nan_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_nan_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_nan_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_nan_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_nan_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_nan_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_nan_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmin_ftz_nan_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_nan_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_nan_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_nan_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_nan_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_nan_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_nan_xorsign_abs_f16 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_nan_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_fmax_ftz_nan_xorsign_abs_f16x2 requires native half type support.
// CHECK_ERROR: error: __nvvm_ldg_h requires native half type support.
// CHECK_ERROR: error: __nvvm_ldg_h2 requires native half type support.
// CHECK_ERROR: error: __nvvm_ldu_h requires native half type support.
// CHECK_ERROR: error: __nvvm_ldu_h2 requires native half type support.

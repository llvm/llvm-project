/*===---- __clang_hip_libdevice_declares.h - HIP device library decls -------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_HIP_LIBDEVICE_DECLARES_H__
#define __CLANG_HIP_LIBDEVICE_DECLARES_H__

#if !defined(__HIPCC_RTC__) && __has_include("hip/hip_version.h")
#include "hip/hip_version.h"
#endif // __has_include("hip/hip_version.h")

#ifdef __cplusplus
extern "C" {
#endif

// BEGIN FLOAT
__device__ __attribute__((const)) float __ocml_acos_f32(float);
__device__ __attribute__((pure)) float __ocml_acosh_f32(float);
__device__ __attribute__((const)) float __ocml_asin_f32(float);
__device__ __attribute__((pure)) float __ocml_asinh_f32(float);
__device__ __attribute__((const)) float __ocml_atan2_f32(float, float);
__device__ __attribute__((const)) float __ocml_atan_f32(float);
__device__ __attribute__((pure)) float __ocml_atanh_f32(float);
__device__ __attribute__((pure)) float __ocml_cbrt_f32(float);
__device__ float __ocml_cos_f32(float);
__device__ float __ocml_native_cos_f32(float);
__device__ __attribute__((pure)) __device__ float __ocml_cosh_f32(float);
__device__ float __ocml_cospi_f32(float);
__device__ float __ocml_i0_f32(float);
__device__ float __ocml_i1_f32(float);
__device__ __attribute__((pure)) float __ocml_erfc_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcinv_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcx_f32(float);
__device__ __attribute__((pure)) float __ocml_erf_f32(float);
__device__ __attribute__((pure)) float __ocml_erfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_expm1_f32(float);
__device__ __attribute__((const)) float __ocml_fdim_f32(float, float);
__device__ __attribute__((const)) __device__ float __ocml_fmod_f32(float,
                                                                   float);
__device__ float __ocml_frexp_f32(float,
                                  __attribute__((address_space(5))) int *);
__device__ __attribute__((const)) float __ocml_hypot_f32(float, float);
__device__ __attribute__((const)) int __ocml_ilogb_f32(float);
__device__ float __ocml_j0_f32(float);
__device__ float __ocml_j1_f32(float);
__device__ float __ocml_lgamma_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_log1p_f32(float);
__device__ __attribute__((const)) float __ocml_logb_f32(float);
__device__ float __ocml_modf_f32(float,
                                 __attribute__((address_space(5))) float *);
__device__ __attribute__((const)) float __ocml_nextafter_f32(float, float);
__device__ __attribute__((const)) float __ocml_len3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_len4_f32(float, float, float,
                                                        float);
__device__ __attribute__((pure)) float __ocml_ncdf_f32(float);
__device__ __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_pow_f32(float, float);
__device__ __attribute__((pure)) float __ocml_pown_f32(float, int);
__device__ __attribute__((pure)) float __ocml_rcbrt_f32(float);
__device__ __attribute__((const)) float __ocml_remainder_f32(float, float);
__device__ float __ocml_remquo_f32(float, float,
                                   __attribute__((address_space(5))) int *);
__device__ __attribute__((const)) float __ocml_rhypot_f32(float, float);
__device__ __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_rlen4_f32(float, float, float,
                                                         float);
__device__ __attribute__((pure)) float __ocml_rsqrt_f32(float);
__device__ float __ocml_sincos_f32(float,
                                   __attribute__((address_space(5))) float *);
__device__ float __ocml_sincospi_f32(float,
                                     __attribute__((address_space(5))) float *);
__device__ float __ocml_sin_f32(float);
__device__ float __ocml_native_sin_f32(float);
__device__ __attribute__((pure)) float __ocml_sinh_f32(float);
__device__ float __ocml_sinpi_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_f32(float);
__device__ float __ocml_tan_f32(float);
__device__ __attribute__((pure)) float __ocml_tanh_f32(float);
__device__ float __ocml_tgamma_f32(float);
__device__ float __ocml_y0_f32(float);
__device__ float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__ __attribute__((const)) float __ocml_add_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__device__ __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__ __attribute__((const)) double __ocml_acos_f64(double);
__device__ __attribute__((pure)) double __ocml_acosh_f64(double);
__device__ __attribute__((const)) double __ocml_asin_f64(double);
__device__ __attribute__((pure)) double __ocml_asinh_f64(double);
__device__ __attribute__((const)) double __ocml_atan2_f64(double, double);
__device__ __attribute__((const)) double __ocml_atan_f64(double);
__device__ __attribute__((pure)) double __ocml_atanh_f64(double);
__device__ __attribute__((pure)) double __ocml_cbrt_f64(double);
__device__ double __ocml_cos_f64(double);
__device__ __attribute__((pure)) double __ocml_cosh_f64(double);
__device__ double __ocml_cospi_f64(double);
__device__ double __ocml_i0_f64(double);
__device__ double __ocml_i1_f64(double);
__device__ __attribute__((pure)) double __ocml_erfc_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcinv_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcx_f64(double);
__device__ __attribute__((pure)) double __ocml_erf_f64(double);
__device__ __attribute__((pure)) double __ocml_erfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_exp10_f64(double);
__device__ __attribute__((pure)) double __ocml_exp2_f64(double);
__device__ __attribute__((pure)) double __ocml_exp_f64(double);
__device__ __attribute__((pure)) double __ocml_expm1_f64(double);
__device__ __attribute__((const)) double __ocml_fdim_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmod_f64(double, double);
__device__ __attribute__((const)) double __ocml_hypot_f64(double, double);
__device__ __attribute__((const)) int __ocml_ilogb_f64(double);
__device__ double __ocml_j0_f64(double);
__device__ double __ocml_j1_f64(double);
__device__ double __ocml_lgamma_f64(double);
__device__ __attribute__((pure)) double __ocml_log10_f64(double);
__device__ __attribute__((pure)) double __ocml_log1p_f64(double);
__device__ __attribute__((pure)) double __ocml_log2_f64(double);
__device__ __attribute__((const)) double __ocml_logb_f64(double);
__device__ __attribute__((pure)) double __ocml_log_f64(double);
__device__ double __ocml_modf_f64(double,
                                  __attribute__((address_space(5))) double *);
__device__ __attribute__((const)) double __ocml_nextafter_f64(double, double);
__device__ __attribute__((const)) double __ocml_len3_f64(double, double,
                                                         double);
__device__ __attribute__((const)) double __ocml_len4_f64(double, double, double,
                                                         double);
__device__ __attribute__((pure)) double __ocml_ncdf_f64(double);
__device__ __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_pow_f64(double, double);
__device__ __attribute__((pure)) double __ocml_pown_f64(double, int);
__device__ __attribute__((pure)) double __ocml_rcbrt_f64(double);
__device__ __attribute__((const)) double __ocml_remainder_f64(double, double);
__device__ double __ocml_remquo_f64(double, double,
                                    __attribute__((address_space(5))) int *);
__device__ __attribute__((const)) double __ocml_rhypot_f64(double, double);
__device__ __attribute__((const)) double __ocml_rlen3_f64(double, double,
                                                          double);
__device__ __attribute__((const)) double __ocml_rlen4_f64(double, double,
                                                          double, double);
__device__ __attribute__((pure)) double __ocml_rsqrt_f64(double);
__device__ double __ocml_sincos_f64(double,
                                    __attribute__((address_space(5))) double *);
__device__ double
__ocml_sincospi_f64(double, __attribute__((address_space(5))) double *);
__device__ double __ocml_sin_f64(double);
__device__ __attribute__((pure)) double __ocml_sinh_f64(double);
__device__ double __ocml_sinpi_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_f64(double);
__device__ double __ocml_tan_f64(double);
__device__ __attribute__((pure)) double __ocml_tanh_f64(double);
__device__ double __ocml_tgamma_f64(double);
__device__ double __ocml_y0_f64(double);
__device__ double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__ __attribute__((const)) double __ocml_add_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__device__ __attribute__((const)) double __ocml_fma_rte_f64(double, double,
                                                            double);
__device__ __attribute__((const)) double __ocml_fma_rtn_f64(double, double,
                                                            double);
__device__ __attribute__((const)) double __ocml_fma_rtp_f64(double, double,
                                                            double);
__device__ __attribute__((const)) double __ocml_fma_rtz_f64(double, double,
                                                            double);

__device__ _Float16 __ocml_cos_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
__device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
__device__ _Float16 __ocml_sin_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);

typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
typedef short __2i16 __attribute__((ext_vector_type(2)));

// We need to match C99's bool and get an i1 in the IR.
#ifdef __cplusplus
typedef bool __ockl_bool;
#else
typedef _Bool __ockl_bool;
#endif

__device__ __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b,
                                                     float c, __ockl_bool s);

#if HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR >= 560
#define __DEPRECATED_SINCE_HIP_560(X) __attribute__((deprecated(X)))
#else
#define __DEPRECATED_SINCE_HIP_560(X)
#endif

// Deprecated, should be removed when rocm releases using it are no longer
// relevant.
__DEPRECATED_SINCE_HIP_560("use ((_Float16)1.0) / ")
__device__ inline _Float16 __llvm_amdgcn_rcp_f16(_Float16 x) {
  return ((_Float16)1.0f) / x;
}

__DEPRECATED_SINCE_HIP_560("use ((__2f16)1.0) / ")
__device__ inline __2f16
__llvm_amdgcn_rcp_2f16(__2f16 __x)
{
  return ((__2f16)1.0f) / __x;
}

#undef __DEPRECATED_SINCE_HIP_560

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __CLANG_HIP_LIBDEVICE_DECLARES_H__

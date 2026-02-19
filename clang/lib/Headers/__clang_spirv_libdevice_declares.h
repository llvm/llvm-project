/*===-- __clang_spirv_libdevice_declares.h - decls for libdevice functions --===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_SPIRV_LIBDEVICE_DECLARES_H__
#define __CLANG_SPIRV_LIBDEVICE_DECLARES_H__

#if defined(__cplusplus)
extern "C" {
#endif

#define _CLC_OVERLOAD [[clang::overloadable]]
#define _CLC_CONSTFN [[gnu::const]]
_CLC_OVERLOAD _CLC_CONSTFN unsigned int __spirv_ocl_s_abs(int);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_acos(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_acos(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_acosh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_acosh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_asin(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_asin(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_asinh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_asinh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_atan(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_atan(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_atan2(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_atan2(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_atanh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_atanh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_cbrt(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_cbrt(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_ceil(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_ceil(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_cos(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_cos(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_cosh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_cosh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_cospi(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_cospi(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_erf(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_erf(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_erfc(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_erfc(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_exp(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_exp(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_exp2(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_exp2(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_exp10(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_exp10(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_expm1(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_expm1(double);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsNan(float);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsNan(double);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsInf(float);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsInf(double);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsFinite(float);
_CLC_OVERLOAD _CLC_CONSTFN bool __spirv_IsFinite(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_copysign(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_copysign(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_ldexp(float, int);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_ldexp(double, int);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fabs(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fabs(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_logb(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_logb(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fmax(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fmax(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fmin(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fmin(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fdim(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fdim(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_floor(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_floor(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fma(float, float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fma(double, double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_fmod(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_fmod(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_frexp(float, int *);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_frexp(double, int *);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_hypot(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_hypot(double, double);
_CLC_OVERLOAD _CLC_CONSTFN int __spirv_ocl_ilogb(float);
_CLC_OVERLOAD _CLC_CONSTFN int __spirv_ocl_ilogb(double);
_CLC_OVERLOAD _CLC_CONSTFN unsigned long __spirv_ocl_s_abs(long);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_lgamma(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_lgamma(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_round(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_round(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_log(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_log(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_log10(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_log10(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_log1p(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_log1p(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_log2(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_log2(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_modf(float, float *);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_modf(double, double *);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_nextafter(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_nextafter(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_sqrt(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_sqrt(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_rsqrt(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_rsqrt(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_pow(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_pow(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_pown(float, int);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_pown(double, int);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_remainder(float, float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_remainder(double, double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_remquo(float, float, int *);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_remquo(double, double, int *);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_sin(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_sin(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_sincos(float, float *);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_sincos(double, double *);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_sinh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_sinh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_sinpi(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_sinpi(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_tan(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_tan(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_tanh(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_tanh(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_tgamma(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_tgamma(double);
_CLC_OVERLOAD _CLC_CONSTFN float __spirv_ocl_trunc(float);
_CLC_OVERLOAD _CLC_CONSTFN double __spirv_ocl_trunc(double);

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // __CLANG_SPIRV_LIBDEVICE_DECLARES_H__

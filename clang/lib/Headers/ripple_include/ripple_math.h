
//===------------- ripple_math.h: Math functions for ripple --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define sqrtf16(x) __builtin_ripple_sqrtf16(x)
#define sqrtf(x) __builtin_ripple_sqrtf(x)
#define sqrt(x) __builtin_ripple_sqrt(x)
#define sqrtl(x) __builtin_ripple_sqrtl(x)

#define asinf16(x) __builtin_ripple_asinf16(x)
#define asinf(x) __builtin_ripple_asinf(x)
#define asin(x) __builtin_ripple_asin(x)
#define asinl(x) __builtin_ripple_asinl(x)

#define acosf16(x) __builtin_ripple_acosf16(x)
#define acosf(x) __builtin_ripple_acosf(x)
#define acos(x) __builtin_ripple_acos(x)
#define acosl(x) __builtin_ripple_acosl(x)

#define atanf16(x) __builtin_ripple_atanf16(x)
#define atanf(x) __builtin_ripple_atanf(x)
#define atan(x) __builtin_ripple_atan(x)
#define atanl(x) __builtin_ripple_atanl(x)

#define atan2f16(x, y) __builtin_ripple_atan2f16((x), (y))
#define atan2f(x, y) __builtin_ripple_atan2f((x), (y))
#define atan2(x, y) __builtin_ripple_atan2((x), (y))
#define atan2l(x, y) __builtin_ripple_atan2l((x), (y))

#define sinf16(x) __builtin_ripple_sinf16(x)
#define sinf(x) __builtin_ripple_sinf(x)
#define sin(x) __builtin_ripple_sin(x)
#define sinl(x) __builtin_ripple_sinl(x)

#define cosf16(x) __builtin_ripple_cosf16(x)
#define cosf(x) __builtin_ripple_cosf(x)
#define cos(x) __builtin_ripple_cos(x)
#define cosl(x) __builtin_ripple_cosl(x)

#define tanf16(x) __builtin_ripple_tanf16(x)
#define tanf(x) __builtin_ripple_tanf(x)
#define tan(x) __builtin_ripple_tan(x)
#define tanl(x) __builtin_ripple_tanl(x)

#define sinhf16(x) __builtin_ripple_sinhf16(x)
#define sinhf(x) __builtin_ripple_sinhf(x)
#define sinh(x) __builtin_ripple_sinh(x)
#define sinhl(x) __builtin_ripple_sinhl(x)

#define coshf16(x) __builtin_ripple_coshf16(x)
#define coshf(x) __builtin_ripple_coshf(x)
#define cosh(x) __builtin_ripple_cosh(x)
#define coshl(x) __builtin_ripple_coshl(x)

#define tanhf16(x) __builtin_ripple_tanhf16(x)
#define tanhf(x) __builtin_ripple_tanhf(x)
#define tanh(x) __builtin_ripple_tanh(x)
#define tanhl(x) __builtin_ripple_tanhl(x)

#define powf16(x, y) __builtin_ripple_powf16((x), (y))
#define powf(x, y) __builtin_ripple_powf((x), (y))
#define pow(x, y) __builtin_ripple_pow((x), (y))
#define powl(x, y) __builtin_ripple_powl((x), (y))

#define logf16(x) __builtin_ripple_logf16(x)
#define logf(x) __builtin_ripple_logf(x)
#define log(x) __builtin_ripple_log(x)
#define logl(x) __builtin_ripple_logl(x)

#define log10f16(x) __builtin_ripple_log10f16(x)
#define log10f(x) __builtin_ripple_log10f(x)
#define log10(x) __builtin_ripple_log10(x)
#define log10l(x) __builtin_ripple_log10l(x)

#define log2f16(x) __builtin_ripple_log2f16(x)
#define log2f(x) __builtin_ripple_log2f(x)
#define log2(x) __builtin_ripple_log2(x)
#define log2l(x) __builtin_ripple_log2l(x)

#define expf16(x) __builtin_ripple_expf16(x)
#define expf(x) __builtin_ripple_expf(x)
#define exp(x) __builtin_ripple_exp(x)
#define expl(x) __builtin_ripple_expl(x)

#define exp2f16(x) __builtin_ripple_exp2f16(x)
#define exp2f(x) __builtin_ripple_exp2f(x)
#define exp2(x) __builtin_ripple_exp2(x)
#define exp2l(x) __builtin_ripple_exp2l(x)

#define exp10f16(x) __builtin_ripple_exp10f16(x)
#define exp10f(x) __builtin_ripple_exp10f(x)
#define exp10(x) __builtin_ripple_exp10(x)
#define exp10l(x) __builtin_ripple_exp10l(x)

#define fabsf16(x) __builtin_ripple_fabsf16(x)
#define fabsf(x) __builtin_ripple_fabsf(x)
#define fabs(x) __builtin_ripple_fabs(x)
#define fabsl(x) __builtin_ripple_fabsl(x)

#define copysignf16(x, y) __builtin_ripple_copysignf16((x), (y))
#define copysignf(x, y) __builtin_ripple_copysignf((x), (y))
#define copysign(x, y) __builtin_ripple_copysign((x), (y))
#define copysignl(x, y) __builtin_ripple_copysignl((x), (y))

#define floorf16(x) __builtin_ripple_floorf16(x)
#define floorf(x) __builtin_ripple_floorf(x)
#define floor(x) __builtin_ripple_floor(x)
#define floorl(x) __builtin_ripple_floorl(x)

#define ceilf16(x) __builtin_ripple_ceilf16(x)
#define ceilf(x) __builtin_ripple_ceilf(x)
#define ceil(x) __builtin_ripple_ceil(x)
#define ceill(x) __builtin_ripple_ceill(x)

#define truncf16(x) __builtin_ripple_truncf16(x)
#define truncf(x) __builtin_ripple_truncf(x)
#define trunc(x) __builtin_ripple_trunc(x)
#define truncl(x) __builtin_ripple_truncl(x)

#define rintf16(x) __builtin_ripple_rintf16(x)
#define rintf(x) __builtin_ripple_rintf(x)
#define rint(x) __builtin_ripple_rint(x)
#define rintl(x) __builtin_ripple_rintl(x)

#define nearbyintf16(x) __builtin_ripple_nearbyintf16(x)
#define nearbyintf(x) __builtin_ripple_nearbyintf(x)
#define nearbyint(x) __builtin_ripple_nearbyint(x)
#define nearbyintl(x) __builtin_ripple_nearbyintl(x)

#define roundf16(x) __builtin_ripple_roundf16(x)
#define roundf(x) __builtin_ripple_roundf(x)
#define round(x) __builtin_ripple_round(x)
#define roundl(x) __builtin_ripple_roundl(x)

#define roundevenf16(x) __builtin_ripple_roundevenf16(x)
#define roundevenf(x) __builtin_ripple_roundevenf(x)
#define roundeven(x) __builtin_ripple_roundeven(x)
#define roundevenl(x) __builtin_ripple_roundevenl(x)

#define ldexpf16(x, y) __builtin_ripple_ldexpf16((x), (y))
#define ldexpf(x, y) __builtin_ripple_ldexpf((x), (y))
#define ldexp(x, y) __builtin_ripple_ldexp((x), (y))
#define ldexpl(x, y) __builtin_ripple_ldexpl((x), (y))

/* ___________________________ isnan/isinf/isfinite __________________________*/

// Use the definitions exposed by the targets in clang/lib/Basic/Targets/*.cpp
// to infer _Float16 support.
// Ripple only supports ARM/AArch64/X86[_64]/Hexagon for now
#if (defined(__hexagon__) && __HVX_ARCH__ >= 68) ||                            \
    __ARM_FEATURE_FP16_SCALAR_ARITHMETIC || __arm64ec__ || __aarch64__ ||      \
    __SSE2__

#define isnan(val)                                                             \
  _Generic((val),                                                              \
      int8_t: 0,                                                               \
      int16_t: 0,                                                              \
      int32_t: 0,                                                              \
      int64_t: 0,                                                              \
      uint8_t: 0,                                                              \
      uint16_t: 0,                                                             \
      uint32_t: 0,                                                             \
      uint64_t: 0,                                                             \
      _Float16: __builtin_isnan(val),                                          \
      float: __builtin_isnan(val),                                             \
      double: __builtin_isnan(val))

#define isinf(val)                                                             \
  _Generic((val),                                                              \
      int8_t: 0,                                                               \
      int16_t: 0,                                                              \
      int32_t: 0,                                                              \
      int64_t: 0,                                                              \
      uint8_t: 0,                                                              \
      uint16_t: 0,                                                             \
      uint32_t: 0,                                                             \
      uint64_t: 0,                                                             \
      _Float16: __builtin_isinf(val),                                          \
      float: __builtin_isinf(val),                                             \
      double: __builtin_isinf(val))

#define isfinite(val)                                                          \
  _Generic((val),                                                              \
      int8_t: 1,                                                               \
      int16_t: 1,                                                              \
      int32_t: 1,                                                              \
      int64_t: 1,                                                              \
      uint8_t: 1,                                                              \
      uint16_t: 1,                                                             \
      uint32_t: 1,                                                             \
      uint64_t: 1,                                                             \
      _Float16: __builtin_isfinite(val),                                       \
      float: __builtin_isfinite(val),                                          \
      double: __builtin_isfinite(val))

#else

#define isnan(val)                                                             \
  _Generic((val),                                                              \
      int8_t: 0,                                                               \
      int16_t: 0,                                                              \
      int32_t: 0,                                                              \
      int64_t: 0,                                                              \
      uint8_t: 0,                                                              \
      uint16_t: 0,                                                             \
      uint32_t: 0,                                                             \
      uint64_t: 0,                                                             \
      float: __builtin_isnan(val),                                             \
      double: __builtin_isnan(val))

#define isinf(val)                                                             \
  _Generic((val),                                                              \
      int8_t: 0,                                                               \
      int16_t: 0,                                                              \
      int32_t: 0,                                                              \
      int64_t: 0,                                                              \
      uint8_t: 0,                                                              \
      uint16_t: 0,                                                             \
      uint32_t: 0,                                                             \
      uint64_t: 0,                                                             \
      float: __builtin_isinf(val),                                             \
      double: __builtin_isinf(val))

#define isfinite(val)                                                          \
  _Generic((val),                                                              \
      int8_t: 1,                                                               \
      int16_t: 1,                                                              \
      int32_t: 1,                                                              \
      int64_t: 1,                                                              \
      uint8_t: 1,                                                              \
      uint16_t: 1,                                                             \
      uint32_t: 1,                                                             \
      uint64_t: 1,                                                             \
      float: __builtin_isfinite(val),                                          \
      double: __builtin_isfinite(val))

#endif

#if __STDCPP_BFLOAT16_T__ || __ARM_FEATURE_BF16 || __SSE2__ || __AVX10_2__ ||  \
    (defined(__hexagon__) && __HVX_ARCH__ >= 81)
#define __has_bf16__ 1
#else
#define __has_bf16__ 0
#endif

#if defined(__hexagon__)
#define __has_soft_bf16__ 1
#else
#define __has_soft_bf16__ 0
#endif

#if __has_bf16__
#if __has_soft_bf16__
#define emulate_spec_unary_bf16_mathfn(op)                                     \
  __attribute__((always_inline)) static __bf16 op##bf16(const __bf16 Val) {    \
    return (__bf16)__builtin_ripple_##op##f((float)(Val));                     \
  }

#define emulate_spec_binary_bf16_mathfn(op)                                    \
  __attribute__((always_inline)) static __bf16 op##bf16(const __bf16 X,        \
                                                        const __bf16 Y) {      \
    return (__bf16)__builtin_ripple_##op##f((float)(X), (float)(Y));           \
  }
#else // !__has_soft_bf16__
#define emulate_spec_unary_bf16_mathfn(op)                                     \
  __attribute__((always_inline)) static __bf16 op##bf16(const __bf16 Val) {    \
    return __builtin_ripple_##op##bf16(Val);                                   \
  }
#define emulate_spec_binary_bf16_mathfn(op)                                    \
  __attribute__((always_inline)) static __bf16 op##bf16(const __bf16 X,        \
                                                        const __bf16 Y) {      \
    return __builtin_ripple_##op##bf16(X, Y);                                  \
  }
#endif // __has_soft_bf16__

emulate_spec_unary_bf16_mathfn(sqrt);
emulate_spec_unary_bf16_mathfn(asin);
emulate_spec_unary_bf16_mathfn(acos);
emulate_spec_unary_bf16_mathfn(atan);
emulate_spec_binary_bf16_mathfn(atan2);
emulate_spec_unary_bf16_mathfn(sin);
emulate_spec_unary_bf16_mathfn(cos);
emulate_spec_unary_bf16_mathfn(tan);
emulate_spec_unary_bf16_mathfn(sinh);
emulate_spec_unary_bf16_mathfn(cosh);
emulate_spec_unary_bf16_mathfn(tanh);
emulate_spec_binary_bf16_mathfn(pow);
emulate_spec_unary_bf16_mathfn(log);
emulate_spec_unary_bf16_mathfn(log10);
emulate_spec_unary_bf16_mathfn(log2);
emulate_spec_unary_bf16_mathfn(exp);
emulate_spec_unary_bf16_mathfn(exp10);
emulate_spec_unary_bf16_mathfn(exp2);
emulate_spec_unary_bf16_mathfn(fabs);
emulate_spec_binary_bf16_mathfn(copysign);
emulate_spec_unary_bf16_mathfn(floor);
emulate_spec_unary_bf16_mathfn(ceil);
emulate_spec_unary_bf16_mathfn(trunc);
emulate_spec_unary_bf16_mathfn(rint);
emulate_spec_unary_bf16_mathfn(nearbyint);
emulate_spec_unary_bf16_mathfn(round);
emulate_spec_unary_bf16_mathfn(roundeven);
emulate_spec_binary_bf16_mathfn(ldexp);
#endif // __has_bf16__
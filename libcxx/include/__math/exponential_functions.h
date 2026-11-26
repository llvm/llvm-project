//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_EXPONENTIAL_FUNCTIONS_H
#define _LIBCPP___MATH_EXPONENTIAL_FUNCTIONS_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/promote.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

// exp

inline float exp(float __x) _NOEXCEPT { return __builtin_expf(__x); }

template <class = int>
double exp(double __x) _NOEXCEPT {
  return __builtin_exp(__x);
}

inline long double exp(long double __x) _NOEXCEPT { return __builtin_expl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double exp(_A1 __x) _NOEXCEPT {
  return __builtin_exp((double)__x);
}

// frexp

inline float frexp(float __x, int* __e) _NOEXCEPT { return __builtin_frexpf(__x, __e); }

template <class = int>
double frexp(double __x, int* __e) _NOEXCEPT {
  return __builtin_frexp(__x, __e);
}

inline long double frexp(long double __x, int* __e) _NOEXCEPT { return __builtin_frexpl(__x, __e); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double frexp(_A1 __x, int* __e) _NOEXCEPT {
  return __builtin_frexp((double)__x, __e);
}

// ldexp

inline float ldexp(float __x, int __e) _NOEXCEPT { return __builtin_ldexpf(__x, __e); }

template <class = int>
double ldexp(double __x, int __e) _NOEXCEPT {
  return __builtin_ldexp(__x, __e);
}

inline long double ldexp(long double __x, int __e) _NOEXCEPT { return __builtin_ldexpl(__x, __e); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double ldexp(_A1 __x, int __e) _NOEXCEPT {
  return __builtin_ldexp((double)__x, __e);
}

// exp2

inline float exp2(float __x) _NOEXCEPT { return __builtin_exp2f(__x); }

template <class = int>
double exp2(double __x) _NOEXCEPT {
  return __builtin_exp2(__x);
}

inline long double exp2(long double __x) _NOEXCEPT { return __builtin_exp2l(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double exp2(_A1 __x) _NOEXCEPT {
  return __builtin_exp2((double)__x);
}

// expm1

inline float expm1(float __x) _NOEXCEPT { return __builtin_expm1f(__x); }

template <class = int>
double expm1(double __x) _NOEXCEPT {
  return __builtin_expm1(__x);
}

inline long double expm1(long double __x) _NOEXCEPT { return __builtin_expm1l(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double expm1(_A1 __x) _NOEXCEPT {
  return __builtin_expm1((double)__x);
}

// scalbln

inline float scalbln(float __x, long __y) _NOEXCEPT { return __builtin_scalblnf(__x, __y); }

template <class = int>
double scalbln(double __x, long __y) _NOEXCEPT {
  return __builtin_scalbln(__x, __y);
}

inline long double scalbln(long double __x, long __y) _NOEXCEPT { return __builtin_scalblnl(__x, __y); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double scalbln(_A1 __x, long __y) _NOEXCEPT {
  return __builtin_scalbln((double)__x, __y);
}

// scalbn

inline float scalbn(float __x, int __y) _NOEXCEPT { return __builtin_scalbnf(__x, __y); }

template <class = int>
double scalbn(double __x, int __y) _NOEXCEPT {
  return __builtin_scalbn(__x, __y);
}

inline long double scalbn(long double __x, int __y) _NOEXCEPT { return __builtin_scalbnl(__x, __y); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double scalbn(_A1 __x, int __y) _NOEXCEPT {
  return __builtin_scalbn((double)__x, __y);
}

// pow

inline float pow(float __x, float __y) _NOEXCEPT { return __builtin_powf(__x, __y); }

template <class = int>
double pow(double __x, double __y) _NOEXCEPT {
  return __builtin_pow(__x, __y);
}

inline long double pow(long double __x, long double __y) _NOEXCEPT { return __builtin_powl(__x, __y); }

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
inline __promote_t<_A1, _A2> pow(_A1 __x, _A2 __y) _NOEXCEPT {
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value), "");
  return __math::pow((__result_type)__x, (__result_type)__y);
}

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_EXPONENTIAL_FUNCTIONS_H

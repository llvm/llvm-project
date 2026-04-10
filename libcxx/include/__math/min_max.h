//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_MIN_MAX_H
#define _LIBCPP___MATH_MIN_MAX_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/promote.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

// fmax

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI float fmax(float __x, float __y) _NOEXCEPT {
  return __builtin_fmaxf(__x, __y);
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI double fmax(double __x, double __y) _NOEXCEPT {
  return __builtin_fmax(__x, __y);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long double fmax(long double __x, long double __y) _NOEXCEPT {
  return __builtin_fmaxl(__x, __y);
}

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI __promote_t<_A1, _A2> fmax(_A1 __x, _A2 __y) _NOEXCEPT {
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value), "");
  return __math::fmax((__result_type)__x, (__result_type)__y);
}

// fmin

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI float fmin(float __x, float __y) _NOEXCEPT {
  return __builtin_fminf(__x, __y);
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI double fmin(double __x, double __y) _NOEXCEPT {
  return __builtin_fmin(__x, __y);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long double fmin(long double __x, long double __y) _NOEXCEPT {
  return __builtin_fminl(__x, __y);
}

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI __promote_t<_A1, _A2> fmin(_A1 __x, _A2 __y) _NOEXCEPT {
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value), "");
  return __math::fmin((__result_type)__x, (__result_type)__y);
}

#if _LIBCPP_STD_VER >= 26

// fminimum (IEEE 754-2019 minimum)

// Fallback implementation for all floating-point types
template <typename _Tp>
inline _LIBCPP_HIDE_FROM_ABI constexpr _Tp __fminimum_fallback(_Tp __x, _Tp __y) _NOEXCEPT {
  // Handle NaN: propagate NaN
  if (__builtin_isnan(__x))
    return __x;
  if (__builtin_isnan(__y))
    return __y;

  // Handle signed zeros: -0.0 < +0.0
  if (__x == _Tp(0) && __y == _Tp(0)) {
    const bool __x_is_neg = __builtin_signbit(__x);
    const bool __y_is_neg = __builtin_signbit(__y);
    if (__x_is_neg != __y_is_neg)
      return __x_is_neg ? __x : __y;
  }

  // Regular comparison
  return __x < __y ? __x : __y;
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr float fminimum(float __x, float __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fminimumf) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fminimumf(__x, __y);
#  else
  return __fminimum_fallback(__x, __y);
#  endif
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr double fminimum(double __x, double __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fminimum) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fminimum(__x, __y);
#  else
  return __fminimum_fallback(__x, __y);
#  endif
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr long double
fminimum(long double __x, long double __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fminimuml) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fminimuml(__x, __y);
#  else
  return __fminimum_fallback(__x, __y);
#  endif
}

// fminimum_num (IEEE 754-2019 minimumNumber)

// Fallback implementation for all floating-point types
template <typename _Tp>
inline _LIBCPP_HIDE_FROM_ABI constexpr _Tp __fminimum_num_fallback(_Tp __x, _Tp __y) _NOEXCEPT {
  // Handle NaN: favor non-NaN values
  const bool __x_is_nan = __builtin_isnan(__x);
  const bool __y_is_nan = __builtin_isnan(__y);
  if (__x_is_nan)
    return __y_is_nan ? __x : __y;
  if (__y_is_nan)
    return __x;

  // Handle signed zeros: -0.0 < +0.0
  if (__x == _Tp(0) && __y == _Tp(0)) {
    const bool __x_is_neg = __builtin_signbit(__x);
    const bool __y_is_neg = __builtin_signbit(__y);
    if (__x_is_neg != __y_is_neg)
      return __x_is_neg ? __x : __y;
  }

  // Regular comparison
  return __x < __y ? __x : __y;
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr float fminimum_num(float __x, float __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fminimum_numf) - builtins are currently broken
  return __builtin_fminimum_numf(__x, __y);
#  else
  return __fminimum_num_fallback(__x, __y);
#  endif
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr double fminimum_num(double __x, double __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fminimum_num) - builtins are currently broken
  return __builtin_fminimum_num(__x, __y);
#  else
  return __fminimum_num_fallback(__x, __y);
#  endif
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr long double
fminimum_num(long double __x, long double __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fminimum_numl) - builtins are currently broken
  return __builtin_fminimum_numl(__x, __y);
#  else
  return __fminimum_num_fallback(__x, __y);
#  endif
}

// fmaximum (IEEE 754-2019 maximum)

// Fallback implementation for all floating-point types
template <typename _Tp>
inline _LIBCPP_HIDE_FROM_ABI constexpr _Tp __fmaximum_fallback(_Tp __x, _Tp __y) _NOEXCEPT {
  // Handle NaN: propagate NaN
  if (__builtin_isnan(__x))
    return __x;
  if (__builtin_isnan(__y))
    return __y;

  // Handle signed zeros: -0.0 < +0.0, so max returns +0.0
  if (__x == _Tp(0) && __y == _Tp(0)) {
    const bool __x_is_neg = __builtin_signbit(__x);
    const bool __y_is_neg = __builtin_signbit(__y);
    if (__x_is_neg != __y_is_neg)
      return __x_is_neg ? __y : __x; // Return the positive zero
  }

  // Regular comparison
  return __x > __y ? __x : __y;
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr float fmaximum(float __x, float __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fmaximumf) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fmaximumf(__x, __y);
#  else
  return __fmaximum_fallback(__x, __y);
#  endif
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr double fmaximum(double __x, double __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fmaximum) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fmaximum(__x, __y);
#  else
  return __fmaximum_fallback(__x, __y);
#  endif
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr long double
fmaximum(long double __x, long double __y) _NOEXCEPT {
#  if __has_builtin(__builtin_fmaximuml) && !(defined(__ARM_ARCH) && __ARM_ARCH == 7)
  return __builtin_fmaximuml(__x, __y);
#  else
  return __fmaximum_fallback(__x, __y);
#  endif
}

// fmaximum_num (IEEE 754-2019 maximumNumber)

// Fallback implementation for all floating-point types
template <typename _Tp>
inline _LIBCPP_HIDE_FROM_ABI constexpr _Tp __fmaximum_num_fallback(_Tp __x, _Tp __y) _NOEXCEPT {
  // Handle NaN: favor non-NaN values
  const bool __x_is_nan = __builtin_isnan(__x);
  const bool __y_is_nan = __builtin_isnan(__y);
  if (__x_is_nan)
    return __y_is_nan ? __x : __y;
  if (__y_is_nan)
    return __x;

  // Handle signed zeros: -0.0 < +0.0, so max returns +0.0
  if (__x == _Tp(0) && __y == _Tp(0)) {
    const bool __x_is_neg = __builtin_signbit(__x);
    const bool __y_is_neg = __builtin_signbit(__y);
    if (__x_is_neg != __y_is_neg)
      return __x_is_neg ? __y : __x; // Return the positive zero
  }

  // Regular comparison
  return __x > __y ? __x : __y;
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr float fmaximum_num(float __x, float __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fmaximum_numf) - builtins are currently broken
  return __builtin_fmaximum_numf(__x, __y);
#  else
  return __fmaximum_num_fallback(__x, __y);
#  endif
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr double fmaximum_num(double __x, double __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fmaximum_num) - builtins are currently broken
  return __builtin_fmaximum_num(__x, __y);
#  else
  return __fmaximum_num_fallback(__x, __y);
#  endif
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI constexpr long double
fmaximum_num(long double __x, long double __y) _NOEXCEPT {
#  if 0 // TODO: __has_builtin(__builtin_fmaximum_numl) - builtins are currently broken
  return __builtin_fmaximum_numl(__x, __y);
#  else
  return __fmaximum_num_fallback(__x, __y);
#  endif
}

#endif // _LIBCPP_STD_VER >= 26

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_MIN_MAX_H

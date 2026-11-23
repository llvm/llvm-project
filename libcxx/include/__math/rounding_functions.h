//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_ROUNDING_FUNCTIONS_H
#define _LIBCPP___MATH_ROUNDING_FUNCTIONS_H

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

// ceil

[[__nodiscard__]] inline float ceil(float __x) _NOEXCEPT { return __builtin_ceilf(__x); }

template <class = int>
[[__nodiscard__]] double ceil(double __x) _NOEXCEPT {
  return __builtin_ceil(__x);
}

[[__nodiscard__]] inline long double ceil(long double __x) _NOEXCEPT { return __builtin_ceill(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double ceil(_A1 __x) _NOEXCEPT {
  return __builtin_ceil((double)__x);
}

// floor

[[__nodiscard__]] inline float floor(float __x) _NOEXCEPT { return __builtin_floorf(__x); }

template <class = int>
[[__nodiscard__]] double floor(double __x) _NOEXCEPT {
  return __builtin_floor(__x);
}

[[__nodiscard__]] inline long double floor(long double __x) _NOEXCEPT { return __builtin_floorl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double floor(_A1 __x) _NOEXCEPT {
  return __builtin_floor((double)__x);
}

// llrint

inline long long llrint(float __x) _NOEXCEPT { return __builtin_llrintf(__x); }

template <class = int>
long long llrint(double __x) _NOEXCEPT {
  return __builtin_llrint(__x);
}

inline long long llrint(long double __x) _NOEXCEPT { return __builtin_llrintl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline long long llrint(_A1 __x) _NOEXCEPT {
  return __builtin_llrint((double)__x);
}

// llround

inline long long llround(float __x) _NOEXCEPT { return __builtin_llroundf(__x); }

template <class = int>
long long llround(double __x) _NOEXCEPT {
  return __builtin_llround(__x);
}

inline long long llround(long double __x) _NOEXCEPT { return __builtin_llroundl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline long long llround(_A1 __x) _NOEXCEPT {
  return __builtin_llround((double)__x);
}

// lrint

inline long lrint(float __x) _NOEXCEPT { return __builtin_lrintf(__x); }

template <class = int>
long lrint(double __x) _NOEXCEPT {
  return __builtin_lrint(__x);
}

inline long lrint(long double __x) _NOEXCEPT { return __builtin_lrintl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline long lrint(_A1 __x) _NOEXCEPT {
  return __builtin_lrint((double)__x);
}

// lround

inline long lround(float __x) _NOEXCEPT { return __builtin_lroundf(__x); }

template <class = int>
long lround(double __x) _NOEXCEPT {
  return __builtin_lround(__x);
}

inline long lround(long double __x) _NOEXCEPT { return __builtin_lroundl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline long lround(_A1 __x) _NOEXCEPT {
  return __builtin_lround((double)__x);
}

// nearbyint

[[__nodiscard__]] inline float nearbyint(float __x) _NOEXCEPT { return __builtin_nearbyintf(__x); }

template <class = int>
[[__nodiscard__]] double nearbyint(double __x) _NOEXCEPT {
  return __builtin_nearbyint(__x);
}

[[__nodiscard__]] inline long double nearbyint(long double __x) _NOEXCEPT { return __builtin_nearbyintl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double nearbyint(_A1 __x) _NOEXCEPT {
  return __builtin_nearbyint((double)__x);
}

// nextafter

inline float nextafter(float __x, float __y) _NOEXCEPT { return __builtin_nextafterf(__x, __y); }

template <class = int>
double nextafter(double __x, double __y) _NOEXCEPT {
  return __builtin_nextafter(__x, __y);
}

inline long double nextafter(long double __x, long double __y) _NOEXCEPT { return __builtin_nextafterl(__x, __y); }

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
inline __promote_t<_A1, _A2> nextafter(_A1 __x, _A2 __y) _NOEXCEPT {
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value), "");
  return __math::nextafter((__result_type)__x, (__result_type)__y);
}

// nexttoward

inline float nexttoward(float __x, long double __y) _NOEXCEPT { return __builtin_nexttowardf(__x, __y); }

template <class = int>
double nexttoward(double __x, long double __y) _NOEXCEPT {
  return __builtin_nexttoward(__x, __y);
}

inline long double nexttoward(long double __x, long double __y) _NOEXCEPT { return __builtin_nexttowardl(__x, __y); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline double nexttoward(_A1 __x, long double __y) _NOEXCEPT {
  return __builtin_nexttoward((double)__x, __y);
}

// rint

[[__nodiscard__]] inline float rint(float __x) _NOEXCEPT { return __builtin_rintf(__x); }

template <class = int>
[[__nodiscard__]] double rint(double __x) _NOEXCEPT {
  return __builtin_rint(__x);
}

[[__nodiscard__]] inline long double rint(long double __x) _NOEXCEPT { return __builtin_rintl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double rint(_A1 __x) _NOEXCEPT {
  return __builtin_rint((double)__x);
}

// round

[[__nodiscard__]] inline float round(float __x) _NOEXCEPT { return __builtin_round(__x); }

template <class = int>
[[__nodiscard__]] double round(double __x) _NOEXCEPT {
  return __builtin_round(__x);
}

[[__nodiscard__]] inline long double round(long double __x) _NOEXCEPT { return __builtin_roundl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double round(_A1 __x) _NOEXCEPT {
  return __builtin_round((double)__x);
}

// trunc

[[__nodiscard__]] inline float trunc(float __x) _NOEXCEPT { return __builtin_trunc(__x); }

template <class = int>
[[__nodiscard__]] double trunc(double __x) _NOEXCEPT {
  return __builtin_trunc(__x);
}

[[__nodiscard__]] inline long double trunc(long double __x) _NOEXCEPT { return __builtin_truncl(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline double trunc(_A1 __x) _NOEXCEPT {
  return __builtin_trunc((double)__x);
}

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_ROUNDING_FUNCTIONS_H

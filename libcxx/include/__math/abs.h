//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_ABS_H
#define _LIBCPP___MATH_ABS_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/integer_traits.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

// fabs

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI float fabs(float __x) _NOEXCEPT { return __builtin_fabsf(__x); }

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI double fabs(double __x) _NOEXCEPT {
  return __builtin_fabs(__x);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long double fabs(long double __x) _NOEXCEPT {
  return __builtin_fabsl(__x);
}

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI double fabs(_A1 __x) _NOEXCEPT {
  return __builtin_fabs((double)__x);
}

// abs

[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline float abs(float __x) _NOEXCEPT { return __builtin_fabsf(__x); }
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline double abs(double __x) _NOEXCEPT { return __builtin_fabs(__x); }

[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline long double abs(long double __x) _NOEXCEPT {
  return __builtin_fabsl(__x);
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline int abs(int __x) _NOEXCEPT {
  return __builtin_abs(__x);
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline long abs(long __x) _NOEXCEPT {
  return __builtin_labs(__x);
}

template <class = int>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI inline long long abs(long long __x) _NOEXCEPT {
  return __builtin_llabs(__x);
}

// Overload for __int128 and signed _BitInt(N) where sizeof(_Tp) >=
// sizeof(int). The sizeof check excludes shorter signed types (signed
// short, signed _BitInt(16)). The standard overloads catch the standard
// types via integer promotion. _BitInt narrower than int does not
// promote and is intentionally unsupported by std::abs.
//
// The explicit !is_same exclusions for int/long/long long are
// load-bearing: signed _BitInt(33..64) shares sizeof with long long, so
// a sizeof-only gate would let those types compete with the existing
// builtin-based overloads.
template <class _Tp,
          __enable_if_t<__is_signed_integer_v<_Tp> && !is_same<_Tp, int>::value && !is_same<_Tp, long>::value &&
                            !is_same<_Tp, long long>::value && (sizeof(_Tp) >= sizeof(int)),
                        int> = 0>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _Tp abs(_Tp __x) _NOEXCEPT {
  return __x < 0 ? -__x : __x;
}

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_ABS_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_GAMMA_H
#define _LIBCPP___MATH_GAMMA_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

// lgamma

inline _LIBCPP_HIDE_FROM_ABI float lgamma(float __x) _NOEXCEPT { return __builtin_lgammaf(__x); }

template <class = int>
_LIBCPP_HIDE_FROM_ABI double lgamma(double __x) _NOEXCEPT {
  return __builtin_lgamma(__x);
}

inline _LIBCPP_HIDE_FROM_ABI long double lgamma(long double __x) _NOEXCEPT { return __builtin_lgammal(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI double lgamma(_A1 __x) _NOEXCEPT {
  return __builtin_lgamma((double)__x);
}

// nan

// tgamma

inline _LIBCPP_HIDE_FROM_ABI float tgamma(float __x) _NOEXCEPT { return __builtin_tgammaf(__x); }

template <class = int>
_LIBCPP_HIDE_FROM_ABI double tgamma(double __x) _NOEXCEPT {
  return __builtin_tgamma(__x);
}

inline _LIBCPP_HIDE_FROM_ABI long double tgamma(long double __x) _NOEXCEPT { return __builtin_tgammal(__x); }

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI double tgamma(_A1 __x) _NOEXCEPT {
  return __builtin_tgamma((double)__x);
}

} // namespace __math

// __lgamma_r

struct __lgamma_result {
  double __result;
  int __sign;
};

#if _LIBCPP_AVAILABILITY_HAS_THREAD_SAFE_LGAMMA
_LIBCPP_EXPORTED_FROM_ABI __lgamma_result __lgamma_thread_safe_impl(double) _NOEXCEPT;

inline _LIBCPP_HIDE_FROM_ABI __lgamma_result __lgamma_thread_safe(double __d) _NOEXCEPT {
  return std::__lgamma_thread_safe_impl(__d);
}
#else
// When deploying to older targets, call `lgamma_r` directly but avoid declaring the actual
// function since different platforms declare the function slightly differently.
double __lgamma_r_shim(double, int*) _NOEXCEPT __asm__("lgamma_r");

inline _LIBCPP_HIDE_FROM_ABI __lgamma_result __lgamma_thread_safe(double __d) _NOEXCEPT {
  int __sign;
  double __res = std::__lgamma_r_shim(__d, &__sign);
  return __lgamma_result{__res, __sign};
}
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_GAMMA_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CMATH_LERP_H
#define _LIBCPP___CMATH_LERP_H

#include <__config>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <__type_traits/promote.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename _Fp>
_LIBCPP_HIDE_FROM_ABI constexpr _Fp __lerp(_Fp __a, _Fp __b, _Fp __t) noexcept {
  if ((__a <= 0 && __b >= 0) || (__a >= 0 && __b <= 0))
    return __t * __b + (1 - __t) * __a;

  if (__t == 1)
    return __b;
  const _Fp __x = __a + __t * (__b - __a);
  if ((__t > 1) == (__b > __a))
    return __b < __x ? __x : __b;
  else
    return __x < __b ? __x : __b;
}

_LIBCPP_HIDE_FROM_ABI inline constexpr float lerp(float __a, float __b, float __t) noexcept {
  return __lerp(__a, __b, __t);
}

_LIBCPP_HIDE_FROM_ABI inline constexpr double lerp(double __a, double __b, double __t) noexcept {
  return __lerp(__a, __b, __t);
}

_LIBCPP_HIDE_FROM_ABI inline constexpr long double lerp(long double __a, long double __b, long double __t) noexcept {
  return __lerp(__a, __b, __t);
}

template <class _A1, class _A2, class _A3>
  requires(is_arithmetic_v<_A1> && is_arithmetic_v<_A2> && is_arithmetic_v<_A3>)
_LIBCPP_HIDE_FROM_ABI inline constexpr __promote_t<_A1, _A2, _A3> lerp(_A1 __a, _A2 __b, _A3 __t) noexcept {
  using __result_type = __promote_t<_A1, _A2, _A3>;
  static_assert(!(
      _IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value && _IsSame<_A3, __result_type>::value));
  return std::__lerp((__result_type)__a, (__result_type)__b, (__result_type)__t);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___CMATH_LERP_H

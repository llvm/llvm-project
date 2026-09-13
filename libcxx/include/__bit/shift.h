//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BIT_SHIFT_H
#define _LIBCPP___BIT_SHIFT_H

#include <__config>
#include <__type_traits/integer_traits.h>
#include <__type_traits/make_unsigned.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <__signed_or_unsigned_integer _Tp, __signed_or_unsigned_integer _Shift>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp shl(_Tp __t, _Shift __cnt) noexcept {
  constexpr auto __n = static_cast<_Shift>(numeric_limits<make_unsigned_t<_Tp>>::digits);
  if constexpr (__is_signed_integer_v<_Shift>) {
    if (__cnt < 0) {
      if (__cnt <= -__n) {
        if constexpr (__is_signed_integer_v<_Tp>)
          return static_cast<_Tp>(__t < 0 ? -1 : 0);
        else
          return static_cast<_Tp>(0);
      }
      return __t >> -__cnt;
    }
  }
  if (__cnt >= __n)
    return static_cast<_Tp>(0);
  return __t << __cnt;
}

template <__signed_or_unsigned_integer _Tp, __signed_or_unsigned_integer _Shift>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp shr(_Tp __t, _Shift __cnt) noexcept {
  constexpr auto __n = static_cast<_Shift>(numeric_limits<make_unsigned_t<_Tp>>::digits);
  if constexpr (__is_signed_integer_v<_Shift>) {
    if (__cnt < 0) {
      if (__cnt <= -__n)
        return static_cast<_Tp>(0);
      return __t << -__cnt;
    }
  }
  if (__cnt >= __n) {
    if constexpr (__is_signed_integer_v<_Tp>)
      return static_cast<_Tp>(__t < 0 ? -1 : 0);
    else
      return static_cast<_Tp>(0);
  }
  return __t >> __cnt;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___BIT_SHIFT_H

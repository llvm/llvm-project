//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BIT_ROTATE_H
#define _LIBCPP___BIT_ROTATE_H

#include <__config>
#include <__type_traits/integer_traits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Writing two full functions for rotl and rotr makes it easier for the compiler
// to optimize the code. On x86 this function becomes the ROL instruction and
// the rotr function becomes the ROR instruction.

#if _LIBCPP_STD_VER >= 20

template <__unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp rotl(_Tp __t, int __cnt) noexcept {
  const int __n = numeric_limits<_Tp>::digits;
  int __r       = __cnt % __n;

  if (__r == 0)
    return __t;

  if (__r > 0)
    return (__t << __r) | (__t >> (__n - __r));

  return (__t >> -__r) | (__t << (__n + __r));
}

template <__unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp rotr(_Tp __t, int __cnt) noexcept {
  const int __n = numeric_limits<_Tp>::digits;
  int __r       = __cnt % __n;

  if (__r == 0)
    return __t;

  if (__r > 0)
    return (__t >> __r) | (__t << (__n - __r));

  return (__t << -__r) | (__t >> (__n + __r));
}

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___BIT_ROTATE_H

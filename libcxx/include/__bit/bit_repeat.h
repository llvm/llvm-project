//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BIT_BIT_REPEAT_H
#define _LIBCPP___BIT_BIT_REPEAT_H

#include <__config>
#include <__type_traits/integer_traits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <__unsigned_integer _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr _Tp bit_repeat(_Tp __t, int __l) {
  _Tp __res = 0;
  for (int __i = 0; __i < numeric_limits<_Tp>::digits; ++__i) {
    __res |= ((__t >> (__i % __l)) & 1) << __i;
  }
  return __res;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___BIT_BIT_REPEAT_H

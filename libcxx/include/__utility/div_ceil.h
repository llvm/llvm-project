//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_DIV_CEIL_H
#define _LIBCPP___UTILITY_DIV_CEIL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 11

// __div_ceil computes the ceiling of the division of two integers. It is mainly used by
// range adaptors like chunk_view and stride_view.

template <class _Integral>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto __div_ceil(_Integral __num, _Integral __denom) {
  _Integral __r = __num / __denom;
  if (__num % __denom)
    ++__r;
  return __r;
}

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_DIV_CEIL_H

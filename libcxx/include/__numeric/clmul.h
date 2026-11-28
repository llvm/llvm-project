//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CLMUL_H
#define _LIBCPP___CLMUL_H

#include <__config>
#include <__numeric/wide_result.h>
#include <__type_traits/integer_traits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <__unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp clmul(_Tp __x, _Tp __y) noexcept {
  return __builtin_elementwise_clmul(__x, __y);
}

template <__unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr wide_result<_Tp> widening_clmul(_Tp __x, _Tp __y) noexcept {
  using _Wide           = wide_result<_Tp>::_Wide;
  constexpr int __width = numeric_limits<_Tp>::digits;

  if constexpr (!is_same_v<_Wide, void>) {
    _Wide __result = __builtin_elementwise_clmul(static_cast<_Wide>(__x), static_cast<_Wide>(__y));
    return wide_result<_Tp>(static_cast<_Tp>(__result >> __width), static_cast<_Tp>(__result));
  } else {
    // This could be simplified if Clang added support for the GCC-style
    // __builtin_clmulg(type a, type b, type *c), where the high bits of the
    // product are written to c.
    constexpr int __low_width = __width / 2;
    constexpr _Tp __low_mask  = (static_cast<_Tp>(1) << __low_width) - 1;
    _Tp __x_low               = __x & __low_mask;
    _Tp __y_low               = __y & __low_mask;
    _Tp __x_high              = __x >> __low_width;
    _Tp __y_high              = __y >> __low_width;

    _Tp __low       = __builtin_elementwise_clmul(__x, __y);
    _Tp __high_low  = __builtin_elementwise_clmul(__x_low, __y_high) ^ __builtin_elementwise_clmul(__x_high, __y_low);
    _Tp __high_high = __builtin_elementwise_clmul(__x_high, __y_high);

    _Tp __high = static_cast<_Tp>(__high_high ^ (__high_low >> (__width - __low_width)));
    return wide_result<_Tp>(__high, __low);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___CLMUL_H

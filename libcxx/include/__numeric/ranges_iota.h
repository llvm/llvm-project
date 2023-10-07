// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NUMERIC_RANGES_IOTA_H
#define _LIBCPP___NUMERIC_RANGES_IOTA_H

#include <__algorithm/out_value_result.h>
#include <__config>
#include <__ranges/concepts.h>
#include <__utility/as_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23
namespace ranges {
template <typename _Out, typename _Tp>
using iota_result = ranges::out_value_result<_Out, _Tp>;

struct __iota_fn {
  template <input_or_output_iterator _Out, sentinel_for<_Out> _Sent, weakly_incrementable _Tp>
    requires indirectly_writable<_Out, const _Tp&>
  constexpr iota_result<_Out, _Tp> operator()(_Out __first, _Sent __last, _Tp __value) const {
    while (__first != __last) {
      *__first = std::as_const(__value);
      ++__first;
      ++__value;
    }
    return {std::move(__first), std::move(__value)};
  }

  template <weakly_incrementable _Tp, ranges::output_range<const _Tp&> _Range>
  constexpr iota_result<ranges::borrowed_iterator_t<_Range>, _Tp> operator()(_Range&& __r, _Tp __value) const {
    return (*this)(ranges::begin(__r), ranges::end(__r), std::move(__value));
  }
};

inline constexpr __iota_fn iota{};
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NUMERIC_RANGES_IOTA_H

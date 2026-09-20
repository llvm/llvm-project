//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_SHIFT_RIGHT_H
#define _LIBCPP___ALGORITHM_RANGES_SHIFT_RIGHT_H

#include <__algorithm/iterator_operations.h>
#include <__algorithm/shift_right.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/permutable.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/subrange.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {
namespace __shift_right {

struct __fn {
  template <permutable _Iter, sentinel_for<_Iter> _Sent>
  _LIBCPP_HIDE_FROM_ABI static constexpr subrange<_Iter>
  operator()(_Iter __first, _Sent __last, iter_difference_t<_Iter> __n) {
    auto __ret = std::__shift_right<_RangeAlgPolicy>(std::move(__first), std::move(__last), std::move(__n));
    return {std::move(__ret.first), std::move(__ret.second)};
  }

  template <forward_range _Range>
    requires permutable<iterator_t<_Range>>
  _LIBCPP_HIDE_FROM_ABI static constexpr borrowed_subrange_t<_Range>
  operator()(_Range&& __range, range_difference_t<_Range> __n) {
    if constexpr (sized_range<_Range>) {
      if (__n >= ranges::distance(__range)) {
        auto __iter = ranges::begin(__range);
        auto __end  = ranges::end(__range);
        ranges::advance(__iter, __end);
        return {__iter, std::move(__iter)};
      }
    }

    auto __ret = std::__shift_right<_RangeAlgPolicy>(ranges::begin(__range), ranges::end(__range), std::move(__n));
    return {std::move(__ret.first), std::move(__ret.second)};
  }
};
} // namespace __shift_right

inline namespace __cpo {
inline constexpr auto shift_right = __shift_right::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_SHIFT_RIGHT_H

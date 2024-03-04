//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H
#define _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H

#include <__algorithm/iterator_operations.h>
#include <__algorithm/shift_left.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/permutable.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/subrange.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __shift_left {

struct __fn {
  template <permutable _Iter, sentinel_for<_Iter> _Sent>
  _LIBCPP_HIDE_FROM_ABI static constexpr subrange<_Iter>
  operator()(_Iter __first, _Sent __last, iter_difference_t<_Iter> __n) {
    auto __ret = std::__shift_left<_RangeAlgPolicy>(std::move(__first), std::move(__last), std::move(__n));
    return {std::move(__ret.first), std::move(__ret.second)};
  }

  template <forward_range _Range>
    requires permutable<iterator_t<_Range>>
  _LIBCPP_HIDE_FROM_ABI static constexpr borrowed_subrange_t<_Range>
  operator()(_Range&& __range, range_difference_t<_Range> __n) {
    auto __ret = std::__shift_left<_RangeAlgPolicy>(ranges::begin(__range), ranges::end(__range), std::move(__n));
    return {std::move(__ret.first), std::move(__ret.second)};
  }
};

} // namespace __shift_left

inline namespace __cpo {
inline constexpr auto shift_left = __shift_left::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H

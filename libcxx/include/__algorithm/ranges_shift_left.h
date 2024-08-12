//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H
#define _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/subrange.h>
#include <__utility/move.h>
#include <__utility/forward.h>
#include <__iterator/advance.h>
#include <__iterator/distance.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

namespace __shift_left {

struct __fn {
  template <std::permutable _Iter, std::sentinel_for<_Iter> _Sent>
  _LIBCPP_HIDE_FROM_ABI constexpr subrange<_Iter>
  operator()(_Iter __first, _Sent __last, iter_difference_t<_Iter> __n) const {
    if (__n <= 0) {
      return {__first, __first};
    }

    auto __dist = std::ranges::distance(__first, __last);
    if (__n >= __dist) {
      return {__first, __first};
    }

    auto __mid = std::ranges::next(__first, __n);
    auto __new_last = std::move(__mid, __last, __first);
    return {__first, __new_last};
  }

  template <std::ranges::forward_range _Range>
  requires std::permutable<iterator_t<_Range>>
  _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Range>
  operator()(_Range&& __range, range_difference_t<_Range> __n) const {
    return (*this)(std::ranges::begin(__range), std::ranges::end(__range), __n);
  }
};

} // namespace __shift_left

inline namespace __cpo {
  inline constexpr auto shift_left = __shift_left::__fn{};
} // namespace __cpo

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_SHIFT_LEFT_H

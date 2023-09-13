//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_CONTAINS_SUBRANGE_H
#define _LIBCPP___ALGORITHM_RANGES_CONTAINS_SUBRANGE_H

#include <__algorithm/ranges_starts_with.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/ranges_operations.h>
#include <__functional/reference_wrapper.h>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/indirectly_comparable.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __contains_subrange {
struct __fn {
  template <input_iterator _Iter1,
            sentinel_for<_Iter1> _Sent1,
            input_iterator _Iter2,
            sentinel_for<_Iter2> _Sent2,
            class _Pred,
            class _Proj1,
            class _Proj2,
            class _Offset>
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __contains_subrange_fn_impl(
      _Iter1 __first1,
      _Sent1 __last1,
      _Iter2 __first2,
      _Sent2 __last2,
      _Pred& __pred,
      _Proj1& __proj1,
      _Proj2& __proj2,
      _Offset __offset) {
    if (__offset < 0)
      return false;
    else {
      for (; __offset >= 0; __offset--, __first1++) {
        auto result = ranges::starts_with(
            std::move(__first1),
            std::move(__last1),
            std::move(__first2),
            std::move(__last2),
            std::ref(__pred),
            std::ref(__proj1),
            std::ref(__proj2));
        if (result)
          return true;
      }
      return false;
    }
  }

  template <input_iterator _Iter1,
            sentinel_for<_Iter1> _Sent1,
            input_iterator _Iter2,
            sentinel_for<_Iter2> _Sent2,
            class _Pred  = ranges::equal_to,
            class _Proj1 = identity,
            class _Proj2 = identity>
    requires indirectly_comparable<_Iter1, _Iter2, _Pred, _Proj1, _Proj2>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr bool operator()(
      _Iter1 __first1,
      _Sent1 __last1,
      _Iter2 __first2,
      _Sent2 __last2,
      _Pred __pred   = {},
      _Proj1 __proj1 = {},
      _Proj2 __proj2 = {}) const {
    auto __n1     = ranges::distance(__first1, __last1);
    auto __n2     = ranges::distance(__first2, __last2);
    auto __offset = __n1 - __n2;

    return __contains_subrange_fn_impl(
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        std::move(__last2),
        __pred,
        __proj1,
        __proj2,
        std::move(__offset));
  }

  template <input_range _Range1,
            input_range _Range2,
            class _Pred  = ranges::equal_to,
            class _Proj1 = identity,
            class _Proj2 = identity>
    requires indirectly_comparable<iterator_t<_Range1>, iterator_t<_Range2>, _Pred, _Proj1, _Proj2>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr bool operator()(
      _Range1&& __range1, _Range2&& __range2, _Pred __pred = {}, _Proj1 __proj1 = {}, _Proj2 __proj2 = {}) const {
    auto __n1 = 0;
    auto __n2 = 0;

    if constexpr (sized_range<_Range1> && sized_range<_Range2>) {
      __n1 = ranges::size(__range1);
      __n2 = ranges::size(__range2);
    } else {
      __n1 = ranges::distance(ranges::begin(__range1), ranges::end(__range1));
      __n2 = ranges::distance(ranges::begin(__range2), ranges::end(__range2));
    }

    auto __offset = __n1 - __n2;
    return __contains_subrange_fn_impl(
        ranges::begin(__range1),
        ranges::end(__range1),
        ranges::begin(__range2),
        ranges::end(__range2),
        __pred,
        __proj1,
        __proj2,
        __offset);
    }
};
} // namespace __contains_subrange

inline namespace __cpo {
inline constexpr auto contains_subrange = __contains_subrange::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___ALGORITHM_RANGES_CONTAINS_SUBRANGE_H
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_DISTANCE_H
#define _LIBCPP___ITERATOR_DISTANCE_H

#include <__algorithm/for_each_segment.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/size.h>
#include <__type_traits/decay.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20
template <class _Iter>
using __iter_distance_t _LIBCPP_NODEBUG = std::iter_difference_t<_Iter>;
#else
template <class _Iter>
using __iter_distance_t _LIBCPP_NODEBUG = typename iterator_traits<_Iter>::difference_type;
#endif

template <class _InputIter, class _Sent>
inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_CONSTEXPR_SINCE_CXX17 __iter_distance_t<_InputIter> __distance(_InputIter __first, _Sent __last) {
  __iter_distance_t<_InputIter> __r(0);
  for (; __first != __last; ++__first)
    ++__r;
  return __r;
}

template <class _RandIter, __enable_if_t<__has_random_access_iterator_category<_RandIter>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 __iter_distance_t<_RandIter>
__distance(_RandIter __first, _RandIter __last) {
  return __last - __first;
}

#if _LIBCPP_STD_VER >= 20
template <class _SegmentedIter,
          __enable_if_t<!__has_random_access_iterator_category<_SegmentedIter>::value &&
                            __is_segmented_iterator_v<_SegmentedIter>,
                        int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 __iter_distance_t<_SegmentedIter>
__distance(_SegmentedIter __first, _SegmentedIter __last) {
  __iter_distance_t<_SegmentedIter> __r(0);
  std::__for_each_segment(__first, __last, [&__r](auto __lfirst, auto __llast) {
    __r += std::__distance(__lfirst, __llast);
  });
  return __r;
}
#endif // _LIBCPP_STD_VER >= 20

template <class _InputIter>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 typename iterator_traits<_InputIter>::difference_type
distance(_InputIter __first, _InputIter __last) {
  return std::__distance(__first, __last);
}

#if _LIBCPP_STD_VER >= 20

// [range.iter.op.distance]

namespace ranges {
struct __distance {
  template <class _Ip, sentinel_for<_Ip> _Sp>
    requires(!sized_sentinel_for<_Sp, _Ip>)
  _LIBCPP_HIDE_FROM_ABI constexpr iter_difference_t<_Ip> operator()(_Ip __first, _Sp __last) const {
    return std::__distance(std::move(__first), std::move(__last));
  }

  template <class _Ip, sized_sentinel_for<decay_t<_Ip>> _Sp>
  _LIBCPP_HIDE_FROM_ABI constexpr iter_difference_t<_Ip> operator()(_Ip&& __first, _Sp __last) const {
    if constexpr (sized_sentinel_for<_Sp, __remove_cvref_t<_Ip>>) {
      return __last - __first;
    } else {
      return __last - decay_t<_Ip>(__first);
    }
  }

  template <range _Rp>
  _LIBCPP_HIDE_FROM_ABI constexpr range_difference_t<_Rp> operator()(_Rp&& __r) const {
    if constexpr (sized_range<_Rp>) {
      return static_cast<range_difference_t<_Rp>>(ranges::size(__r));
    } else {
      return operator()(ranges::begin(__r), ranges::end(__r));
    }
  }
};

inline namespace __cpo {
inline constexpr auto distance = __distance{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_DISTANCE_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SHIFT_RIGHT_H
#define _LIBCPP___ALGORITHM_SHIFT_RIGHT_H

#include <__algorithm/iterator_operations.h>
#include <__algorithm/move.h>
#include <__algorithm/move_backward.h>
#include <__algorithm/swap_ranges.h>
#include <__assert>
#include <__concepts/derived_from.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__utility/move.h>
#include <__utility/pair.h>
#include <__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _AlgPolicy, class _Iter, class _Sent>
_LIBCPP_HIDE_FROM_ABI constexpr pair<_Iter, _Iter>
__shift_right(_Iter __first, _Sent __last, typename _IterOps<_AlgPolicy>::template __difference_type<_Iter> __n) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__n >= 0, "Providing a negative shift amount to shift_right is UB");
  if (__n == 0) {
    _Iter __end = _IterOps<_AlgPolicy>::next(__first, __last);
    return pair<_Iter, _Iter>(std::move(__first), std::move(__end));
  }

  using _IterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_Iter>;

  if constexpr (derived_from<_IterCategory, random_access_iterator_tag>) {
    _Iter __end = _IterOps<_AlgPolicy>::next(__first, __last);
    auto __size = __end - __first;
    if (__n >= __size) {
      return pair<_Iter, _Iter>(__end, std::move(__end));
    }
    _Iter __m = __first;
    _IterOps<_AlgPolicy>::advance(__m, (__size - __n));
    auto __ret = std::__move_backward<_AlgPolicy>(std::move(__first), std::move(__m), __end);
    return pair<_Iter, _Iter>(std::move(__ret.second), std::move(__end));
  } else if constexpr (derived_from<_IterCategory, bidirectional_iterator_tag>) {
    _Iter __end = _IterOps<_AlgPolicy>::next(__first, __last);
    if constexpr (sized_sentinel_for<_Sent, _Iter>) {
      if (__n >= ranges::distance(__first, __last)) {
        return pair<_Iter, _Iter>(__end, std::move(__end));
      }
    }
    _Iter __m = __end;
    for (; __n > 0; --__n) {
      if (__m == __first) {
        return pair<_Iter, _Iter>(__end, std::move(__end));
      }
      --__m;
    }
    auto __ret = std::__move_backward<_AlgPolicy>(std::move(__first), std::move(__m), __end);
    return pair<_Iter, _Iter>(std::move(__ret.second), std::move(__end));
  } else {
    _Iter __ret = __first;
    for (; __n > 0; --__n) {
      if (__ret == __last) {
        return pair<_Iter, _Iter>(__ret, std::move(__ret));
      }
      ++__ret;
    }

    // We have an __n-element scratch space from __first to __ret.
    // Slide an __n-element window [__trail, __lead) from left to right.
    // We're essentially doing swap_ranges(__first, __ret, __trail, __lead)
    // over and over; but once __lead reaches __end we needn't bother
    // to save the values of elements [__trail, __end).

    auto __trail = __first;
    auto __lead  = __ret;
    while (__trail != __ret) {
      if (__lead == __last) {
        std::__move<_AlgPolicy>(std::move(__first), std::move(__trail), __ret);
        return pair<_Iter, _Iter>(__ret, std::move(__lead));
      }
      ++__trail;
      ++__lead;
    }

    _Iter __mid = __first;
    while (true) {
      if (__lead == __last) {
        __trail = std::__move<_AlgPolicy>(__mid, __ret, __trail).second;
        std::__move<_AlgPolicy>(std::move(__first), std::move(__mid), std::move(__trail));
        return pair<_Iter, _Iter>(__ret, std::move(__lead));
      }
      _IterOps<_AlgPolicy>::iter_swap(__mid, __trail);
      ++__mid;
      ++__trail;
      ++__lead;
      if (__mid == __ret) {
        __mid = __first;
      }
    }
  }
}

template <class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI constexpr _ForwardIterator
shift_right(_ForwardIterator __first,
            _ForwardIterator __last,
            typename iterator_traits<_ForwardIterator>::difference_type __n) {
  return std::__shift_right<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __n).first;
}

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SHIFT_RIGHT_H

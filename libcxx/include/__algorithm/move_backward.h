//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MOVE_BACKWARD_H
#define _LIBCPP___ALGORITHM_MOVE_BACKWARD_H

#include <__algorithm/copy_move_common.h>
#include <__algorithm/iterator_operations.h>
#include <__config>
#include <__type_traits/is_copy_constructible.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy>
struct __move_backward_loop {
  template <class _InIter, class _Sent, class _OutIter>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, _OutIter>
  operator()(_InIter __first, _Sent __last, _OutIter __result) const {
    auto __last_iter          = _IterOps<_AlgPolicy>::next(__first, __last);
    auto __original_last_iter = __last_iter;

    while (__first != __last_iter) {
      *--__result = _IterOps<_AlgPolicy>::__iter_move(--__last_iter);
    }

    return std::make_pair(std::move(__original_last_iter), std::move(__result));
  }
};

struct __move_backward_trivial {
  // At this point, the iterators have been unwrapped so any `contiguous_iterator` has been unwrapped to a pointer.
  template <class _In, class _Out,
            __enable_if_t<__can_lower_move_assignment_to_memmove<_In, _Out>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_In*, _Out*>
  operator()(_In* __first, _In* __last, _Out* __result) const {
    return std::__copy_backward_trivial_impl(__first, __last, __result);
  }
};

template <class _AlgPolicy, class _BidirectionalIterator1, class _Sentinel, class _BidirectionalIterator2>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
pair<_BidirectionalIterator1, _BidirectionalIterator2>
__move_backward(_BidirectionalIterator1 __first, _Sentinel __last, _BidirectionalIterator2 __result) {
  static_assert(std::is_copy_constructible<_BidirectionalIterator1>::value &&
                std::is_copy_constructible<_BidirectionalIterator1>::value, "Iterators must be copy constructible.");

  return std::__dispatch_copy_or_move<_AlgPolicy, __move_backward_loop<_AlgPolicy>, __move_backward_trivial>(
      std::move(__first), std::move(__last), std::move(__result));
}

template <class _BidirectionalIterator1, class _BidirectionalIterator2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20
_BidirectionalIterator2
move_backward(_BidirectionalIterator1 __first, _BidirectionalIterator1 __last,
              _BidirectionalIterator2 __result)
{
  return std::__move_backward<_ClassicAlgPolicy>(
      std::move(__first), std::move(__last), std::move(__result)).second;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_MOVE_BACKWARD_H

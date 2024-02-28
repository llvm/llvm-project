//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SHIFT_LEFT_H
#define _LIBCPP___ALGORITHM_SHIFT_LEFT_H

#include <__algorithm/move.h>
#include <__config>
#include <__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _AlgPolicy, class _Iter, class _Sent>
inline _LIBCPP_HIDE_FROM_ABI constexpr pair<_Iter, _Iter>
__shift_left(_Iter __first, _Sent __last, typename iterator_traits<_Iter>::difference_type __n) {
  _Iter __end = _IterOps<_AlgPolicy>::next(__first, __last);

  if (__n == 0) {
    return {std::move(__first), std::move(__end)};
  }

  _Iter __m = __first;
  if constexpr (__has_random_access_iterator_category<_Iter>::value) {
    if (__n >= __end - __first) {
      return {std::move(__first), std::move(__first)};
    }
    __m += __n;
  } else {
    for (; __n > 0; --__n) {
      if (__m == __end) {
        return {std::move(__first), std::move(__first)};
      }
      ++__m;
    }
  }

  _Iter __result = std::__move<_AlgPolicy>(__m, __end, __first).second;
  return {std::move(__first), std::move(__result)};
}

template <class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI constexpr _ForwardIterator
shift_left(_ForwardIterator __first,
           _ForwardIterator __last,
           typename iterator_traits<_ForwardIterator>::difference_type __n) {
  return std::__shift_left<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __n).second;
}

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SHIFT_LEFT_H

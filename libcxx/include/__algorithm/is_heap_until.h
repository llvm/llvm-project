//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IS_HEAP_UNTIL_H
#define _LIBCPP___ALGORITHM_IS_HEAP_UNTIL_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Compare, class _Iter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__is_heap_until(_Iter __first, _Iter __last, _Compare&& __comp) {
  auto __count = __last - __first;

  if (__count < 2)
    return __last;

  // This points to one past the last parent that has two children.
  // Integer division handles the case where the last parent has a single child.
  auto __parent_last = __first + (__count - 1) / 2;
  auto __child       = __first;
  ++__child;
  for (; __first != __parent_last; ++__first) {
    if (__comp(*__first, *__child))
      return __child;
    ++__child;

    if (__comp(*__first, *__child))
      return __child;
    ++__child;
  }

  // If the heap is even-sized, the last parent has a single child, handled here.
  if (__count % 2 == 0) {
    if (__comp(*__first, *__child))
      return __child;
    ++__child;
  }

  return __child;
}

template <class _RandomAccessIterator, class _Compare>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  return std::__is_heap_until(__first, __last, static_cast<__comp_ref_type<_Compare> >(__comp));
}

template <class _RandomAccessIterator>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  return std::__is_heap_until(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_IS_HEAP_UNTIL_H

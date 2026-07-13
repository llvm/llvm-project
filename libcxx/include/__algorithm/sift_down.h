//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SIFT_DOWN_H
#define _LIBCPP___ALGORITHM_SIFT_DOWN_H

#include <__algorithm/iterator_operations.h>
#include <__assert>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, bool __assume_both_children, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__sift_down(_RandomAccessIterator __first,
            _Compare&& __comp,
            __iterator_difference_type<_RandomAccessIterator> __len,
            __iterator_difference_type<_RandomAccessIterator> __start) {
  using _Ops = _IterOps<_AlgPolicy>;

  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  typedef typename iterator_traits<_RandomAccessIterator>::value_type value_type;
  // left-child of __start is at 2 * __start + 1
  // right-child of __start is at 2 * __start + 2
  difference_type __child = __start;

  if (__len < 2 || (__len - 2) / 2 < __child)
    return;

  __child = 2 * __child + 1;

  if _LIBCPP_CONSTEXPR (__assume_both_children) {
    // right-child exists and is greater than left-child
    __child += __comp(__first[__child], __first[__child + 1]);
  } else if ((__child + 1) < __len && __comp(__first[__child], __first[__child + 1])) {
    // right-child exists and is greater than left-child
    ++__child;
  }

  // check if we are in heap-order
  if (__comp(__first[__child], __first[__start]))
    // we are, __start is larger than its largest child
    return;

  value_type __top(_Ops::__iter_move(__first + __start));
  do {
    // we are not in heap-order, swap the parent with its largest child
    __first[__start] = _Ops::__iter_move(__first + __child);
    __start          = __child;

    if ((__len - 2) / 2 < __child)
      break;

    // recompute the child based off of the updated parent
    __child = 2 * __child + 1;

    if _LIBCPP_CONSTEXPR (__assume_both_children) {
      __child += __comp(__first[__child], __first[__child + 1]);
    } else if ((__child + 1) < __len && __comp(__first[__child], __first[__child + 1])) {
      // right-child exists and is greater than left-child
      ++__child;
    }

    // check if we are in heap-order
  } while (!__comp(__first[__child], __top));
  __first[__start] = std::move(__top);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _RandomAccessIterator __floyd_sift_down(
    _RandomAccessIterator __first,
    _Compare&& __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __len) {
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  _LIBCPP_ASSERT_INTERNAL(__len >= 2, "shouldn't be called unless __len >= 2");

  _RandomAccessIterator __hole    = __first;
  _RandomAccessIterator __child_i = __first;
  difference_type __child         = 0;

  while (true) {
    __child_i += difference_type(__child + 1);
    __child = 2 * __child + 1;

    if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + difference_type(1)))) {
      // right-child exists and is greater than left-child
      ++__child_i;
      ++__child;
    }

    // swap __hole with its largest child
    *__hole = _IterOps<_AlgPolicy>::__iter_move(__child_i);
    __hole  = __child_i;

    // if __hole is now a leaf, we're done
    if (__child > (__len - 2) / 2)
      return __hole;
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SIFT_DOWN_H

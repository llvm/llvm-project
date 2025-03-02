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

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__sift_down(_RandomAccessIterator __first,
            _Compare&& __comp,
            typename iterator_traits<_RandomAccessIterator>::difference_type __len,
            typename iterator_traits<_RandomAccessIterator>::difference_type __start) {
  using _Ops = _IterOps<_AlgPolicy>;

  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  typedef typename iterator_traits<_RandomAccessIterator>::value_type value_type;

  if (__len < 2)
    return;

  // left-child of __start is at 2 * __start + 1
  // right-child of __start is at 2 * __start + 2
  difference_type __child         = 2 * __start + 1;
  _RandomAccessIterator __child_i = __first + __child, __start_i = __first + __start;

  if ((__child + 1) < __len) {
    _RandomAccessIterator __right_i = _Ops::next(__child_i);
    if (__comp(*__child_i, *__right_i)) {
      // right-child exists and is greater than left-child
      __child_i = __right_i;
      ++__child;
    }
  }

  // check if we are in heap-order
  if (__comp(*__child_i, *__start_i))
    // we are, __start is larger than its largest child
    return;

  value_type __top(_Ops::__iter_move(__start_i));
  do {
    // we are not in heap-order, swap the parent with its largest child
    *__start_i = _Ops::__iter_move(__child_i);
    __start_i  = __child_i;

    if ((__len - 2) / 2 < __child)
      break;

    // recompute the child based off of the updated parent
    __child   = 2 * __child + 1;
    __child_i = __first + __child;

    if ((__child + 1) < __len) {
      _RandomAccessIterator __right_i = _Ops::next(__child_i);
      if (__comp(*__child_i, *__right_i)) {
        // right-child exists and is greater than left-child
        __child_i = __right_i;
        ++__child;
      }
    }

    // check if we are in heap-order
  } while (!__comp(*__child_i, __top));
  *__start_i = std::move(__top);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _RandomAccessIterator __floyd_sift_down(
    _RandomAccessIterator __first,
    _Compare&& __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __len) {
  _LIBCPP_ASSERT_INTERNAL(__len > 1, "shouldn't be called unless __len > 1");

  using _Ops = _IterOps<_AlgPolicy>;

  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  difference_type __child      = 1;
  _RandomAccessIterator __hole = __first, __child_i = __first;

  while (true) {
    __child_i += __child;
    __child *= 2;

    if (__child < __len) {
      _RandomAccessIterator __right_i = _Ops::next(__child_i);
      if (__comp(*__child_i, *__right_i)) {
        // right-child exists and is greater than left-child
        __child_i = __right_i;
        ++__child;
      }
    }

    // swap __hole with its largest child
    *__hole = _Ops::__iter_move(__child_i);
    __hole  = __child_i;

    // if __hole is now a leaf, we're done
    if (__child > __len / 2)
      return __hole;
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SIFT_DOWN_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PUSH_HEAP_H
#define _LIBCPP___ALGORITHM_PUSH_HEAP_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/iterator_operations.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__sift_up(_RandomAccessIterator __first, _RandomAccessIterator __bottom, _Compare&& __comp) {
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  using value_type      = typename iterator_traits<_RandomAccessIterator>::value_type;

  difference_type __parent = __bottom - __first;
  _LIBCPP_ASSERT_INTERNAL(__parent > 0, "shouldn't be called unless __bottom - __first > 0");
  __parent                         = (__parent - 1) / 2;
  _RandomAccessIterator __parent_i = __first + __parent;

  if (__comp(*__parent_i, *__bottom)) {
    value_type __t(_IterOps<_AlgPolicy>::__iter_move(__bottom));
    do {
      *__bottom = _IterOps<_AlgPolicy>::__iter_move(__parent_i);
      __bottom  = __parent_i;
      if (__parent == 0)
        break;
      __parent   = (__parent - 1) / 2;
      __parent_i = __first + __parent;
    } while (__comp(*__parent_i, __t));

    *__bottom = std::move(__t);
  }
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp) {
  if (__first != __last)
    std::__sift_up<_AlgPolicy, __comp_ref_type<_Compare> >(std::move(__first), std::move(--__last), __comp);
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  static_assert(std::is_copy_constructible<_RandomAccessIterator>::value, "Iterators must be copy constructible.");
  static_assert(std::is_copy_assignable<_RandomAccessIterator>::value, "Iterators must be copy assignable.");

  std::__push_heap<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  std::push_heap(std::move(__first), std::move(__last), __less<>());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PUSH_HEAP_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MAKE_HEAP_H
#define _LIBCPP___ALGORITHM_MAKE_HEAP_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/iterator_operations.h>
#include <__algorithm/push_heap.h>
#include <__algorithm/sift_down.h>
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
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp) {
  __comp_ref_type<_Compare> __comp_ref = __comp;

  using __diff_t = __iter_diff_t<_RandomAccessIterator>;
  const __diff_t __n   = __last - __first;

  const __diff_t __odd_n = (__n & 1) ? __n : __n - 1;

  static const bool __assume_both_children = is_arithmetic<__iter_value_type<_RandomAccessIterator> >::value;

  if (__n > 1) {
    // start from the first parent, there is no need to consider children

    for (__diff_t __start = (__odd_n - 2) / 2; __start >= 0; --__start) {
      std::__sift_down<_AlgPolicy, __assume_both_children>(__first, __comp_ref, __odd_n, __start);
    }
    if _LIBCPP_CONSTEXPR (__assume_both_children)
      std::__sift_up<_AlgPolicy>(__first, __last, __comp, __n);
  }
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  std::__make_heap<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  std::make_heap(std::move(__first), std::move(__last), __less<>());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_MAKE_HEAP_H

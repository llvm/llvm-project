//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_REPLACE_IF_H
#define _LIBCPP___ALGORITHM_REPLACE_IF_H

#include <__algorithm/for_each.h>
#include <__config>
#include <__functional/identity.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Predicate, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
replace_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, const _Tp& __new_value) {
  auto __apply = [&__pred, &__new_value](auto&& __curr) {
    if (__pred(__curr)) {
      __curr = __new_value;
    }
  };

  // We implement __replace_if using __for_each to inherit its optimizations for
  // segmented iterators. This improves performance without adding complexity.
  __identity __proj;
  std::__for_each(std::move(__first), std::move(__last), __apply, __proj);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_REPLACE_IF_H

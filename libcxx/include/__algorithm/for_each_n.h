// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOR_EACH_N_H
#define _LIBCPP___ALGORITHM_FOR_EACH_N_H

#include <__algorithm/for_each.h>
#include <__algorithm/for_each_n_segment.h>
#include <__algorithm/min.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__iterator/unreachable_sentinel.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_same.h>
#include <__utility/exchange.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class _Size, class _Func, class _Proj>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__for_each_n(_Iter __first, _Sent __last, _Size& __n, _Func& __func, _Proj& __proj) {
  if constexpr (__has_random_access_iterator_category<_Iter>::value) {
    if constexpr (is_same_v<_Sent, __unreachable_sentinel_t>) {
      return std::__for_each(__first, __first + std::exchange(__n, 0), __func, __proj);
    } else {
      auto __count = std::min<size_t>(__n, __last - __first);
      __n -= __count;
      return std::__for_each(__first, __first + __count, __func, __proj);
    }
  } else if constexpr (__is_segmented_iterator_v<_Iter>) {
    return std::__for_each_n_segment(__first, __n, [&](auto __lfirst, auto __llast, _Size __max) {
      std::__for_each_n(__lfirst, __llast, __max, __func, __proj);
      return __max;
    });
  } else {
    while (__n > 0 && __first != __last) {
      std::__invoke(__func, std::__invoke(__proj, *__first));
      ++__first;
      --__n;
    }
    return std::move(__first);
  }
}

template <class _InputIterator, class _Size, class _Func>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
for_each_n(_InputIterator __first, _Size __orig_n, _Func __f) {
  __identity __proj;
  return std::__for_each_n(__first, __unreachable_sentinel, __orig_n, __f, __proj);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FOR_EACH_N_H

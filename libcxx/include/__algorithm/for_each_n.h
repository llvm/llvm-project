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
#include <__algorithm/in_fun_result.h>
#include <__assert>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__memory/replacable_object.h>
#include <__type_traits/invoke.h>
#include <__utility/convert_to_integral.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Size, class _Func, class _Proj>
_LIBCPP_CONSTEXPR_SINCE_CXX20 __in_fun_result<_InputIterator, _Func>
__for_each_n(_InputIterator __first, _Size __n, _Func __func, _Proj& __proj) {
#ifndef _LIBCPP_CXX03_LANG
  if constexpr (__is_segmented_iterator_v<_InputIterator>) {
    using __local_iterator = typename __segmented_iterator_traits<_InputIterator>::__local_iterator;
    if constexpr (__has_random_access_iterator_category<__local_iterator>::value) {
      __replacable_object<_Func> __f(std::move(__func));
      auto __iter = std::__for_each_n_segment(__first, __n, [&](__local_iterator __lfirst, __local_iterator __llast) {
        __f.__replace(std::__for_each(__lfirst, __llast, std::move(__f.__get()), __proj).__fun_);
      });
      return {std::move(__iter), std::move(__f.__get())};
    } else {
      return std::__for_each(__first, __first + __n, std::move(__func), __proj);
    }
  } else
#endif
  {
    while (__n > 0) {
      std::__invoke(__func, std::__invoke(__proj, *__first));
      ++__first;
      --__n;
    }
    return {std::move(__first), std::move(__func)};
  }
}

#if _LIBCPP_STD_VER >= 17

template <class _InputIterator, class _Size, class _Func>
inline _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator for_each_n(_InputIterator __first, _Size __orig_n, _Func __f) {
  auto __n = std::__convert_to_integral(__orig_n);
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__n >= 0, "for_each_n requires a non-negative count");
  __identity __proj;
  return std::__for_each_n(__first, __n, __f, __proj).__in_;
}

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FOR_EACH_N_H

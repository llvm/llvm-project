// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SEARCH_H
#define _LIBCPP___ALGORITHM_SEARCH_H

#include <__config>
#include <__algorithm/comp.h>
#include <__functional/search.h>
#include <__iterator/iterator_traits.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator1
search(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
       _BinaryPredicate __pred) {
  return _VSTD::__search<typename add_lvalue_reference<_BinaryPredicate>::type>(
             __first1, __last1, __first2, __last2, __pred,
             typename iterator_traits<_ForwardIterator1>::iterator_category(),
             typename iterator_traits<_ForwardIterator2>::iterator_category()).first;
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator1
search(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  typedef typename iterator_traits<_ForwardIterator1>::value_type __v1;
  typedef typename iterator_traits<_ForwardIterator2>::value_type __v2;
  return _VSTD::search(__first1, __last1, __first2, __last2, __equal_to<__v1, __v2>());
}

#if _LIBCPP_STD_VER > 14
template <class _ForwardIterator, class _Searcher>
_LIBCPP_NODISCARD_EXT _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator
search(_ForwardIterator __f, _ForwardIterator __l, const _Searcher& __s) {
  return __s(__f, __l).first;
}

#endif

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SEARCH_H

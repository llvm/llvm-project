// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_EQUAL_H
#define _LIBCPP___CXX03___ALGORITHM_EQUAL_H

#include <__cxx03/__algorithm/comp.h>
#include <__cxx03/__algorithm/unwrap_iter.h>
#include <__cxx03/__config>
#include <__cxx03/__functional/identity.h>
#include <__cxx03/__iterator/distance.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__string/constexpr_c_functions.h>
#include <__cxx03/__type_traits/desugars_to.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/invoke.h>
#include <__cxx03/__type_traits/is_equality_comparable.h>
#include <__cxx03/__type_traits/is_volatile.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool __equal_iter_impl(
    _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate& __pred) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      return false;
  return true;
}

template <class _Tp,
          class _Up,
          class _BinaryPredicate,
          __enable_if_t<__desugars_to_v<__equal_tag, _BinaryPredicate, _Tp, _Up> && !is_volatile<_Tp>::value &&
                            !is_volatile<_Up>::value && __libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                        int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool
__equal_iter_impl(_Tp* __first1, _Tp* __last1, _Up* __first2, _BinaryPredicate&) {
  return std::__constexpr_memcmp_equal(__first1, __first2, __element_count(__last1 - __first1));
}

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred) {
  return std::__equal_iter_impl(
      std::__unwrap_iter(__first1), std::__unwrap_iter(__last1), std::__unwrap_iter(__first2), __pred);
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2) {
  return std::equal(__first1, __last1, __first2, __equal_to());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_EQUAL_H

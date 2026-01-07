// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_ITERATOR_H
#define _LIBCPP___ITERATOR_ITERATOR_H

#include <__config>
#include <__cstddef/ptrdiff_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Category, class _Tp, class _Distance = ptrdiff_t, class _Pointer = _Tp*, class _Reference = _Tp&>
struct _LIBCPP_DEPRECATED_IN_CXX17 iterator {
  typedef _Tp value_type;
  typedef _Distance difference_type;
  typedef _Pointer pointer;
  typedef _Reference reference;
  typedef _Category iterator_category;
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
#ifdef _LIBCPP_ABI_NO_ITERATOR_BASES
template <class _Derived>
struct __no_iterator_base {};

template <class _Derived, class _Category, class _Tp, class _Distance, class _Pointer, class _Reference>
using __iterator_base _LIBCPP_NODEBUG = __no_iterator_base<_Derived>;
#else
template <class _Derived, class _Category, class _Tp, class _Distance, class _Pointer, class _Reference>
using __iterator_base _LIBCPP_NODEBUG = iterator<_Category, _Tp, _Distance, _Pointer, _Reference>;
#endif
_LIBCPP_SUPPRESS_DEPRECATED_POP

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_ITERATOR_H

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_ACCESS_H
#define _LIBCPP___CXX03___ITERATOR_ACCESS_H

#include <__cxx03/__config>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, size_t _Np>
_LIBCPP_HIDE_FROM_ABI _Tp* begin(_Tp (&__array)[_Np]) _NOEXCEPT {
  return __array;
}

template <class _Tp, size_t _Np>
_LIBCPP_HIDE_FROM_ABI _Tp* end(_Tp (&__array)[_Np]) _NOEXCEPT {
  return __array + _Np;
}

template <class _Cp>
_LIBCPP_HIDE_FROM_ABI typename _Cp::iterator begin(_Cp& __c) {
  return __c.begin();
}

template <class _Cp>
_LIBCPP_HIDE_FROM_ABI typename _Cp::const_iterator begin(const _Cp& __c) {
  return __c.begin();
}

template <class _Cp>
_LIBCPP_HIDE_FROM_ABI typename _Cp::iterator end(_Cp& __c) {
  return __c.end();
}

template <class _Cp>
_LIBCPP_HIDE_FROM_ABI typename _Cp::const_iterator end(const _Cp& __c) {
  return __c.end();
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_ACCESS_H

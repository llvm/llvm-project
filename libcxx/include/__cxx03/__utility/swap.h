//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___UTILITY_SWAP_H
#define _LIBCPP___CXX03___UTILITY_SWAP_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/is_assignable.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/is_nothrow_assignable.h>
#include <__cxx03/__type_traits/is_nothrow_constructible.h>
#include <__cxx03/__type_traits/is_swappable.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class>
using __swap_result_t = void;

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI __swap_result_t<_Tp> swap(_Tp& __x, _Tp& __y) {
  _Tp __t(std::move(__x));
  __x = std::move(__y);
  __y = std::move(__t);
}

template <class _Tp, size_t _Np, __enable_if_t<__is_swappable_v<_Tp>, int> >
inline _LIBCPP_HIDE_FROM_ABI void swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) {
  for (size_t __i = 0; __i != _Np; ++__i) {
    swap(__a[__i], __b[__i]);
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___UTILITY_SWAP_H

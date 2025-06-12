//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TYPE_TRAITS_IS_SWAPPABLE_H
#define _LIBCPP___CXX03___TYPE_TRAITS_IS_SWAPPABLE_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/add_lvalue_reference.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_assignable.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/is_nothrow_assignable.h>
#include <__cxx03/__type_traits/is_nothrow_constructible.h>
#include <__cxx03/__type_traits/void_t.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Up, class = void>
inline const bool __is_swappable_with_v = false;

template <class _Tp>
inline const bool __is_swappable_v = __is_swappable_with_v<_Tp&, _Tp&>;

template <class _Tp, class _Up, bool = __is_swappable_with_v<_Tp, _Up> >
inline const bool __is_nothrow_swappable_with_v = false;

template <class _Tp>
inline const bool __is_nothrow_swappable_v = __is_nothrow_swappable_with_v<_Tp&, _Tp&>;

template <class>
using __swap_result_t = void;

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI __swap_result_t<_Tp> swap(_Tp& __x, _Tp& __y);

template <class _Tp, size_t _Np, __enable_if_t<__is_swappable_v<_Tp>, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI void swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]);

// ALL generic swap overloads MUST already have a declaration available at this point.

template <class _Tp, class _Up>
inline const bool __is_swappable_with_v<_Tp,
                                        _Up,
                                        __void_t<decltype(swap(std::declval<_Tp>(), std::declval<_Up>())),
                                                 decltype(swap(std::declval<_Up>(), std::declval<_Tp>()))> > = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TYPE_TRAITS_IS_SWAPPABLE_H

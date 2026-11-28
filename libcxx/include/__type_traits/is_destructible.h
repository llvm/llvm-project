//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_DESTRUCTIBLE_H
#define _LIBCPP___TYPE_TRAITS_IS_DESTRUCTIBLE_H

#include <__config>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_destructible : _BoolConstant<__is_destructible(_Tp)> {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_destructible_v = __is_destructible(_Tp);
#endif

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_destructible : integral_constant<bool, __is_nothrow_destructible(_Tp)> {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_destructible_v = __is_nothrow_destructible(_Tp);
#endif

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_trivially_destructible
    : integral_constant<bool, __is_trivially_destructible(_Tp)> {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_trivially_destructible_v = __is_trivially_destructible(_Tp);
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_DESTRUCTIBLE_H

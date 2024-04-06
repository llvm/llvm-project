// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_SPAN_H
#define _LIBCPP___FWD_SPAN_H

#include <__concepts/convertible_to.h>
#include <__concepts/equality_comparable.h>
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_const.h>
#include <cstddef>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

inline constexpr size_t dynamic_extent = numeric_limits<size_t>::max();
template <typename _Tp, size_t _Extent = dynamic_extent>
class span;

#endif

#if _LIBCPP_STD_VER >= 26

template <class _Tp>
concept __integral_constant_like =
    is_integral_v<decltype(_Tp::value)> && !is_same_v<bool, remove_const_t<decltype(_Tp::value)>> &&
    convertible_to<_Tp, decltype(_Tp::value)> && equality_comparable_with<_Tp, decltype(_Tp::value)> &&
    bool_constant<_Tp() == _Tp::value>::value &&
    bool_constant<static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value>::value;

template <class _Tp>
constexpr size_t __maybe_static_ext = dynamic_extent;

template <__integral_constant_like _Tp>
constexpr size_t __maybe_static_ext<_Tp> = {_Tp::value};

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FWD_SPAN_H

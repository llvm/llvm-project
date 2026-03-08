//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_CONSTANT_ARG_H
#define _LIBCPP___UTILITY_CONSTANT_ARG_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <auto _Vp>
struct constant_arg_t {
  _LIBCPP_HIDE_FROM_ABI explicit constant_arg_t() = default;
};

template <auto _Vp>
inline constexpr constant_arg_t<_Vp> constant_arg{};

template <class>
inline constexpr bool __is_constant_arg_t = false;
template <auto _Vp>
inline constexpr bool __is_constant_arg_t<constant_arg_t<_Vp>> = true;

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_CONSTANT_ARG_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___UTILITY_IGNORE_ARGUMENT_H
#define _LIBCPP___UTILITY_IGNORE_ARGUMENT_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 11

_LIBCPP_BEGIN_NAMESPACE_STD

struct __ignore_argument {
  template <typename _Tp>
  constexpr __ignore_argument(_Tp&&) noexcept {}
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 11

#endif // _LIBCPP___UTILITY_IGNORE_ARGUMENT_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::optional<T&>

#ifndef _LIBCPP___FWD_OPTIONAL_H
#define _LIBCPP___FWD_OPTIONAL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class optional;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FWD_OPTIONAL_H

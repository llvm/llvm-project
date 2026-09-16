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

#include <__config>
#include <__cstddef/size_t.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

inline constexpr size_t dynamic_extent = numeric_limits<size_t>::max();
template <typename _Tp, size_t _Extent = dynamic_extent>
class span;

_LIBCPP_END_NAMESPACE_STD

#endif

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FWD_SPAN_H

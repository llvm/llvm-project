//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_VECTOR2_H
#define _LIBCPP___FWD_VECTOR2_H

#include <__config>
#include <__fwd/memory.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

// PROTOTYPE: std::vector2 as proposed by P3866R0. Unlike std::vector, there is intentionally no
// std::vector2<bool> specialization: std::vector2<bool> is a regular container of bool.
template <class _Tp, class _Alloc = allocator<_Tp> >
class vector2;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___FWD_VECTOR2_H

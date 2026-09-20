//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_VECTOR_H
#define _LIBCPP___FWD_VECTOR_H

#include <__config>
#include <__fwd/memory.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Alloc = allocator<_Tp> >
class vector;

template <class _Allocator>
class vector<bool, _Allocator>;

#if _LIBCPP_STD_VER >= 23

template <class _Tp>
inline constexpr bool __is_vector_bool_v = false;

template <class _Allocator>
inline constexpr bool __is_vector_bool_v<vector<bool, _Allocator>> = true;

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_VECTOR_H

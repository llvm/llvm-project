//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___VECTOR_IS_VECTOR_H
#define _LIBCPP___VECTOR_IS_VECTOR_H

#include <__config>
#include <__fwd/vector.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_vectorbool : false_type {};

template <class _Allocator>
struct __is_vectorbool<vector<bool, _Allocator>> : true_type {};

template <class _Allocator>
inline constexpr bool __is_vectorbool_v = __is_vectorbool<_Allocator>::value;

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___VECTOR_VECTOR_H

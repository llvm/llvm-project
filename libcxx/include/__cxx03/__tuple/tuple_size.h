//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TUPLE_TUPLE_SIZE_H
#define _LIBCPP___CXX03___TUPLE_TUPLE_SIZE_H

#include <__cxx03/__config>
#include <__cxx03/__fwd/tuple.h>
#include <__cxx03/__tuple/tuple_types.h>
#include <__cxx03/__type_traits/is_const.h>
#include <__cxx03/__type_traits/is_volatile.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS tuple_size;

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS tuple_size<const _Tp> : public tuple_size<_Tp> {};
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS tuple_size<volatile _Tp> : public tuple_size<_Tp> {};
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS tuple_size<const volatile _Tp> : public tuple_size<_Tp> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TUPLE_TUPLE_SIZE_H

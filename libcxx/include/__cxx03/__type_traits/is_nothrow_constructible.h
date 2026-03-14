//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
#define _LIBCPP___CXX03___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/add_lvalue_reference.h>
#include <__cxx03/__type_traits/add_rvalue_reference.h>
#include <__cxx03/__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template < class _Tp, class... _Args>
struct _LIBCPP_TEMPLATE_VIS is_nothrow_constructible
    : public integral_constant<bool, __is_nothrow_constructible(_Tp, _Args...)> {};

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS is_nothrow_copy_constructible
    : public integral_constant< bool, __is_nothrow_constructible(_Tp, __add_lvalue_reference_t<const _Tp>)> {};

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS is_nothrow_move_constructible
    : public integral_constant<bool, __is_nothrow_constructible(_Tp, __add_rvalue_reference_t<_Tp>)> {};

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS is_nothrow_default_constructible
    : public integral_constant<bool, __is_nothrow_constructible(_Tp)> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H

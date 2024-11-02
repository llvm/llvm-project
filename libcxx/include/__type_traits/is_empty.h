//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_EMPTY_H
#define _LIBCPP___TYPE_TRAITS_IS_EMPTY_H

#include <__config>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_feature(is_empty) || defined(_LIBCPP_COMPILER_GCC)

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS is_empty
    : public integral_constant<bool, __is_empty(_Tp)> {};

#else  // __has_feature(is_empty)

template <class _Tp>
struct __is_empty1
    : public _Tp
{
    double __lx;
};

struct __is_empty2
{
    double __lx;
};

template <class _Tp, bool = is_class<_Tp>::value>
struct __libcpp_empty : public integral_constant<bool, sizeof(__is_empty1<_Tp>) == sizeof(__is_empty2)> {};

template <class _Tp> struct __libcpp_empty<_Tp, false> : public false_type {};

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_empty : public __libcpp_empty<_Tp> {};

#endif // __has_feature(is_empty)

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_empty_v = is_empty<_Tp>::value;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_EMPTY_H

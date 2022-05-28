//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_UNION_H
#define _LIBCPP___TYPE_TRAITS_IS_UNION_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_feature(is_union) || defined(_LIBCPP_COMPILER_GCC)

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_union
    : public integral_constant<bool, __is_union(_Tp)> {};

#else

template <class _Tp> struct __libcpp_union : public false_type {};
template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_union
    : public __libcpp_union<typename remove_cv<_Tp>::type> {};

#endif

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_union_v = is_union<_Tp>::value;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_UNION_H

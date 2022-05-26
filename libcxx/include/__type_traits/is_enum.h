//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_ENUM_H
#define _LIBCPP___TYPE_TRAITS_IS_ENUM_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_feature(is_enum) || defined(_LIBCPP_COMPILER_GCC)

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_enum
    : public integral_constant<bool, __is_enum(_Tp)> {};

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_enum_v = __is_enum(_Tp);
#endif

#else

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_enum
    : public integral_constant<bool, !is_void<_Tp>::value             &&
                                     !is_integral<_Tp>::value         &&
                                     !is_floating_point<_Tp>::value   &&
                                     !is_array<_Tp>::value            &&
                                     !is_pointer<_Tp>::value          &&
                                     !is_reference<_Tp>::value        &&
                                     !is_member_pointer<_Tp>::value   &&
                                     !is_union<_Tp>::value            &&
                                     !is_class<_Tp>::value            &&
                                     !is_function<_Tp>::value         > {};

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_enum_v = is_enum<_Tp>::value;
#endif

#endif // __has_feature(is_enum) || defined(_LIBCPP_COMPILER_GCC)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_ENUM_H

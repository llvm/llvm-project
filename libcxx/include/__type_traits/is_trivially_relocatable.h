//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
#define _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

#include <__config>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/is_trivially_move_constructible.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

//------------------------------------------------------WARNING-------------------------------------------------------//
//
// This type trait should only be used where ABI stability isn't relevant or can be assured by other means,
// like in function bodies or where the difference can be detected at runtime. If this trait is used in ABI-sensitive
// environments it has to be ensured that Clang and GCC can interpret the data produced by each other. If you are not
// certain that this is the case DO NOT USE THIS TYPE TRAIT
//
//--------------------------------------------------------------------------------------------------------------------//

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_trivially_relocatable)

template <class _Type>
struct __libcpp_is_trivially_relocatable : integral_constant<bool, __is_trivially_relocatable(_Type)> {};

#else

template <class _Type>
using __libcpp_is_trivially_relocatable =
    integral_constant<bool, is_trivially_move_constructible<_Type>::value && is_trivially_destructible_v<_Type>::value>;

#endif // __has_builtin(__is_trivially_relocatable)

#if _LIBCPP_STD_VER > 14
template <class _Type>
constexpr bool __libcpp_is_trivially_relocatable_v = __libcpp_is_trivially_relocatable<_Type>::value;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

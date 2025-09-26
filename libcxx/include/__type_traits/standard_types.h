//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_STANDARD_TYPES_H
#define _LIBCPP___TYPE_TRAITS_STANDARD_TYPES_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __is_standard_signed_integer_type_v = false;

template <>
inline constexpr bool __is_standard_signed_integer_type_v<signed char> = true;

template <>
inline constexpr bool __is_standard_signed_integer_type_v<short> = true;

template <>
inline constexpr bool __is_standard_signed_integer_type_v<int> = true;

template <>
inline constexpr bool __is_standard_signed_integer_type_v<long> = true;

template <>
inline constexpr bool __is_standard_signed_integer_type_v<long long> = true;

template <class _Tp>
inline constexpr bool __is_standard_unsigned_integer_type_v = false;

template <>
inline constexpr bool __is_standard_unsigned_integer_type_v<unsigned char> = true;

template <>
inline constexpr bool __is_standard_unsigned_integer_type_v<unsigned short> = true;

template <>
inline constexpr bool __is_standard_unsigned_integer_type_v<unsigned int> = true;

template <>
inline constexpr bool __is_standard_unsigned_integer_type_v<unsigned long> = true;

template <>
inline constexpr bool __is_standard_unsigned_integer_type_v<unsigned long long> = true;

template <class _Tp>
inline constexpr bool __is_standard_integer_type_v =
    __is_standard_signed_integer_type_v<_Tp> || __is_standard_unsigned_integer_type_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_character_type_v = false;

template <>
inline constexpr bool __is_character_type_v<char> = true;

template <>
inline constexpr bool __is_character_type_v<wchar_t> = true;

#if _LIBCPP_HAS_CHAR8_T
template <>
inline constexpr bool __is_character_type_v<char8_t> = true;
#endif

template <>
inline constexpr bool __is_character_type_v<char16_t> = true;

template <>
inline constexpr bool __is_character_type_v<char32_t> = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___TYPE_TRAITS_STANDARD_TYPES_H

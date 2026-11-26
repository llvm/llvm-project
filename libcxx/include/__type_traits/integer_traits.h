//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_INTEGER_TRAITS_H
#define _LIBCPP___TYPE_TRAITS_INTEGER_TRAITS_H

#include <__config>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_signed.h>
#include <__type_traits/is_unqualified.h>
#include <__type_traits/is_unsigned.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// /signed integer type/ and /unsigned integer type/ per [basic.fundamental]
// /p1-2: specific unqualified types plus extended integer types. bool and
// character types are integral but excluded. cv-qualified versions are
// distinct types ([basic.type.qualifier]) and so excluded.

template <class _Tp>
inline const bool __is_character_v = false;
template <>
inline const bool __is_character_v<char> = true;
template <>
inline const bool __is_character_v<wchar_t> = true;
#if _LIBCPP_HAS_CHAR8_T
template <>
inline const bool __is_character_v<char8_t> = true;
#endif
template <>
inline const bool __is_character_v<char16_t> = true;
template <>
inline const bool __is_character_v<char32_t> = true;

template <class _Tp>
inline const bool __is_signed_integer_v =
    is_integral<_Tp>::value && is_signed<_Tp>::value && !__is_character_v<_Tp> && !is_same<_Tp, bool>::value &&
    __is_unqualified_v<_Tp>;

template <class _Tp>
inline const bool __is_unsigned_integer_v =
    is_integral<_Tp>::value && is_unsigned<_Tp>::value && !__is_character_v<_Tp> && !is_same<_Tp, bool>::value &&
    __is_unqualified_v<_Tp>;

#if _LIBCPP_STD_VER >= 20
template <class _Tp>
concept __signed_integer = __is_signed_integer_v<_Tp>;

template <class _Tp>
concept __unsigned_integer = __is_unsigned_integer_v<_Tp>;

// This isn't called __integer, because an integer type according to [basic.fundamental]/p11 is the same as an integral
// type. An integral type is _not_ the same set of types as signed and unsigned integer types combined.
template <class _Tp>
concept __signed_or_unsigned_integer = __signed_integer<_Tp> || __unsigned_integer<_Tp>;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_INTEGER_TRAITS_H

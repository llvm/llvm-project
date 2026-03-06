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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// These traits determine whether a type is a /signed integer type/ or
// /unsigned integer type/ per [basic.fundamental]/p1-2.
//
// Signed/unsigned integer types include the standard types (signed char,
// short, int, long, long long), extended integer types (__int128), and
// bit-precise integer types (_BitInt(N)).
//
// Character types (char, wchar_t, char8_t, char16_t, char32_t) and bool
// are integral but are NOT signed/unsigned integer types.

// clang-format off
template <class _Tp> inline const bool __is_character_or_bool_v = false;
template <> inline const bool __is_character_or_bool_v<bool>     = true;
template <> inline const bool __is_character_or_bool_v<char>     = true;
#if _LIBCPP_HAS_WIDE_CHARACTERS
template <> inline const bool __is_character_or_bool_v<wchar_t>  = true;
#endif
#if _LIBCPP_HAS_CHAR8_T
template <> inline const bool __is_character_or_bool_v<char8_t>  = true;
#endif
template <> inline const bool __is_character_or_bool_v<char16_t> = true;
template <> inline const bool __is_character_or_bool_v<char32_t> = true;
// clang-format on

// Signed integer types: all signed integral types except character types.
// Uses compiler builtins to automatically cover _BitInt(N) for any N.
// CV-qualified types are excluded to match the behavior of the original
// explicit specializations and to avoid accidentally enabling library
// features (e.g. std::formattable) for volatile-qualified types.
template <class _Tp>
inline const bool __is_signed_integer_v =
    !__is_const(_Tp) && !__is_volatile(_Tp) && __is_integral(_Tp) && __is_signed(_Tp) && !__is_character_or_bool_v<_Tp>;

// Unsigned integer types: all unsigned integral types except character types and bool.
template <class _Tp>
inline const bool __is_unsigned_integer_v =
    !__is_const(_Tp) && !__is_volatile(_Tp) && __is_integral(_Tp) && __is_unsigned(_Tp) &&
    !__is_character_or_bool_v<_Tp>;

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

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
#include <__type_traits/integral_constant.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// This trait is to determine whether a type is a /signed integer type/
// See [basic.fundamental]/p1
template <class _Tp>
inline const bool __is_signed_integer_v = false;
template <>
inline const bool __is_signed_integer_v<signed char> = true;
template <>
inline const bool __is_signed_integer_v<signed short> = true;
template <>
inline const bool __is_signed_integer_v<signed int> = true;
template <>
inline const bool __is_signed_integer_v<signed long> = true;
template <>
inline const bool __is_signed_integer_v<signed long long> = true;
#if _LIBCPP_HAS_INT128
template <>
inline const bool __is_signed_integer_v<__int128_t> = true;
#endif

// This trait is to determine whether a type is an /unsigned integer type/
// See [basic.fundamental]/p2
template <class _Tp>
inline const bool __is_unsigned_integer_v = false;
template <>
inline const bool __is_unsigned_integer_v<unsigned char> = true;
template <>
inline const bool __is_unsigned_integer_v<unsigned short> = true;
template <>
inline const bool __is_unsigned_integer_v<unsigned int> = true;
template <>
inline const bool __is_unsigned_integer_v<unsigned long> = true;
template <>
inline const bool __is_unsigned_integer_v<unsigned long long> = true;
#if _LIBCPP_HAS_INT128
template <>
inline const bool __is_unsigned_integer_v<__uint128_t> = true;
#endif

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

// is_integral
// This trait is to determine whether a type is an /integral type/ (a.k.a. /integer type/)
// See [basic.fundamental]/p11

#if __has_builtin(__is_integral)

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_integral : _BoolConstant<__is_integral(_Tp)> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_integral_v = __is_integral(_Tp);
#  endif

#else

template <class _Tp>
inline constexpr bool __is_integral_impl = __is_signed_integer_v<_Tp> || __is_unsigned_integer_v<_Tp>;

template <>
inline constexpr bool __is_integral_impl<bool> = true;

template <>
inline constexpr bool __is_integral_impl<char> = true;

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
inline constexpr bool __is_integral_impl<wchar_t> = true;
#  endif

#  if _LIBCPP_HAS_CHAR8_T
template <>
inline constexpr bool __is_integral_impl<char8_t> = true;
#  endif

template <>
inline constexpr bool __is_integral_impl<char16_t> = true;

template <>
inline constexpr bool __is_integral_impl<char32_t> = true;

template <class _Tp>
struct is_integral : integral_constant<bool, __is_integral_impl<__remove_cv_t<_Tp>>> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
inline constexpr bool is_integral_v = __is_integral_impl<__remove_cv_t<_Tp>>;
#  endif

#endif // __has_builtin(__is_integral)

// is_floating_point
// This trait is to determine whether a type is a /floating-point type/
// See [basic.fundamental]/p12

// FIXME: This should use __is_floating_point once we're able to implement
// numeric_limits for all of them (especially __float128)
template <class>
inline const bool __is_floating_point_impl = false;

template <>
inline const bool __is_floating_point_impl<float> = true;

template <>
inline const bool __is_floating_point_impl<double> = true;

template <>
inline const bool __is_floating_point_impl<long double> = true;

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_floating_point
    : integral_constant<bool, __is_floating_point_impl<__remove_cv_t<_Tp> > > {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_floating_point_v = __is_floating_point_impl<remove_cv_t<_Tp>>;
#endif

// is_arithmetic
// This trait is to determine whether a type is an /arithmetic type/
// See [basic.fundamental]/p14

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_arithmetic
    : integral_constant<bool, is_integral<_Tp>::value || is_floating_point<_Tp>::value> {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_arithmetic_v = is_integral_v<_Tp> || is_floating_point_v<_Tp>;
#endif

// is_signed

#if __has_builtin(__is_signed)

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_signed : _BoolConstant<__is_signed(_Tp)> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_signed_v = __is_signed(_Tp);
#  endif

#else // __has_builtin(__is_signed)

template <class _Tp, bool = is_arithmetic<_Tp>::value>
inline constexpr bool __is_signed_v = false;

template <class _Tp>
inline constexpr bool __is_signed_v<_Tp, true> = _Tp(-1) < _Tp(0);

template <class _Tp>
struct is_signed : integral_constant<bool, __is_signed_v<_Tp>> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
inline constexpr bool is_signed_v = __is_signed_v<_Tp>;
#  endif

#endif // __has_builtin(__is_signed)

// is_unsigned

#if __has_builtin(__is_unsigned)

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_unsigned : _BoolConstant<__is_unsigned(_Tp)> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_unsigned_v = __is_unsigned(_Tp);
#  endif

#else // __has_builtin(__is_unsigned)

template <class _Tp, bool = is_integral<_Tp>::value>
inline constexpr bool __is_unsigned_v = false;

template <class _Tp>
inline constexpr bool __is_unsigned_v<_Tp, true> = _Tp(0) < _Tp(-1);

template <class _Tp>
struct is_unsigned : integral_constant<bool, __is_unsigned_v<_Tp>> {};

#  if _LIBCPP_STD_VER >= 17
template <class _Tp>
inline constexpr bool is_unsigned_v = __is_unsigned_v<_Tp>;
#  endif

#endif // __has_builtin(__is_unsigned)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_INTEGER_TRAITS_H

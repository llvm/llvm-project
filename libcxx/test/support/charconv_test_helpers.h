//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CHARCONV_TEST_HELPERS_H
#define SUPPORT_CHARCONV_TEST_HELPERS_H

#include <cstddef>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#if TEST_STD_VER < 11
#error This file requires C++11
#endif

template <typename To, typename From>
constexpr auto
is_non_narrowing(From a) -> decltype(To{a}, std::true_type())
{
    return {};
}

template <typename To>
constexpr auto
is_non_narrowing(...) -> std::false_type
{
    return {};
}

template <typename X, typename T>
constexpr bool
_fits_in(T, std::true_type /* non-narrowing*/, ...)
{
    return true;
}

template <typename X, typename T, typename xl = std::numeric_limits<X>>
constexpr bool
_fits_in(T v, std::false_type, std::true_type /* T signed*/, std::true_type /* X signed */)
{
    return xl::lowest() <= v && v <= (xl::max)();
}

template <typename X, typename T, typename xl = std::numeric_limits<X>>
constexpr bool
_fits_in(T v, std::false_type, std::true_type /* T signed */, std::false_type /* X unsigned*/)
{
    return 0 <= v && typename std::make_unsigned<T>::type(v) <= (xl::max)();
}

template <typename X, typename T, typename xl = std::numeric_limits<X>>
constexpr bool
_fits_in(T v, std::false_type, std::false_type /* T unsigned */, ...)
{
    return v <= typename std::make_unsigned<X>::type((xl::max)());
}

template <typename X, typename T>
constexpr bool
fits_in(T v)
{
  return _fits_in<X>(v, is_non_narrowing<X>(v), std::is_signed<T>(), std::is_signed<X>());
}

#endif // SUPPORT_CHARCONV_TEST_HELPERS_H

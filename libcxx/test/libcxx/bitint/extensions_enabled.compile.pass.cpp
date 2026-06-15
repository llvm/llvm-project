//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_BITINT_EXTENSIONS

// With the macro on, libc++'s gated facilities accept _BitInt operands.

#include <bit>
#include <charconv>
#include <functional>
#include <type_traits>
#include <utility>

#include "test_macros.h"

#ifdef __BITINT_MAXWIDTH__

static_assert(std::__admits_bitint_extension_v<_BitInt(64)>);
static_assert(std::__admits_bitint_extension_v<unsigned _BitInt(64)>);

template <class T>
concept has_byteswap = requires(T t) { std::byteswap(t); };
template <class T>
concept has_hash = requires(T t) { std::hash<T>{}(t); };
template <class T>
concept has_to_chars = requires(char* b, char* e, T t) { std::to_chars(b, e, t); };
template <class T>
concept has_cmp_less = requires(T t) { std::cmp_less(t, 0); };

using S64 = _BitInt(64);
using U64 = unsigned _BitInt(64);
using U8  = unsigned _BitInt(8);

static_assert(has_byteswap<U64>);
static_assert(has_byteswap<S64>);
static_assert(has_byteswap<U8>);
static_assert(has_hash<U64>);
static_assert(has_hash<S64>);
static_assert(has_to_chars<U64>);
static_assert(has_cmp_less<U64>);

// Type-trait surface is macro-independent (P3666 keeps it).
static_assert(std::is_integral_v<U64>);

#endif // __BITINT_MAXWIDTH__

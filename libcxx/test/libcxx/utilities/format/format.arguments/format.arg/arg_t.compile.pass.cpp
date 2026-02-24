//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// namespace __format { enum class __arg_t : uint8_t{...}; }

#include <format>

#include <cstdint>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_same_v<std::underlying_type_t<std::__format::__arg_t>, std::uint8_t>);

// The 128-bit and 256-bit types are unconditionally in the enum to avoid
// the values depending on the availability of extended integer types.
static_assert(std::uint8_t(std::__format::__arg_t::__none) == 0);
static_assert(std::uint8_t(std::__format::__arg_t::__boolean) == 1);
static_assert(std::uint8_t(std::__format::__arg_t::__char_type) == 2);
static_assert(std::uint8_t(std::__format::__arg_t::__int) == 3);
static_assert(std::uint8_t(std::__format::__arg_t::__long_long) == 4);
static_assert(std::uint8_t(std::__format::__arg_t::__i128) == 5);
static_assert(std::uint8_t(std::__format::__arg_t::__i256) == 6);
static_assert(std::uint8_t(std::__format::__arg_t::__unsigned) == 7);
static_assert(std::uint8_t(std::__format::__arg_t::__unsigned_long_long) == 8);
static_assert(std::uint8_t(std::__format::__arg_t::__u128) == 9);
static_assert(std::uint8_t(std::__format::__arg_t::__u256) == 10);
static_assert(std::uint8_t(std::__format::__arg_t::__float) == 11);
static_assert(std::uint8_t(std::__format::__arg_t::__double) == 12);
static_assert(std::uint8_t(std::__format::__arg_t::__long_double) == 13);
static_assert(std::uint8_t(std::__format::__arg_t::__const_char_type_ptr) == 14);
static_assert(std::uint8_t(std::__format::__arg_t::__string_view) == 15);
static_assert(std::uint8_t(std::__format::__arg_t::__ptr) == 16);
static_assert(std::uint8_t(std::__format::__arg_t::__handle) == 17);

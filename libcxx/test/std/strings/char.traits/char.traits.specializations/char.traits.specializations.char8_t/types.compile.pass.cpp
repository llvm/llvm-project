//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// template<> struct char_traits<char8_t>

// using char_type           = char8_t;
// using int_type            = unsigned int;
// using off_type            = streamoff;
// using pos_type            = u8streampos;
// using state_type          = mbstate_t;
// using comparison_category = strong_ordering;

#include <string>
#include <type_traits>
#include <cstdint>

#include "test_macros.h"

#ifndef TEST_HAS_NO_CHAR8_T
static_assert(std::is_same<std::char_traits<char8_t>::char_type, char8_t>::value);
static_assert(std::is_same<std::char_traits<char8_t>::int_type, unsigned int>::value);
static_assert(std::is_same<std::char_traits<char8_t>::off_type, std::streamoff>::value);
static_assert(std::is_same<std::char_traits<char8_t>::pos_type, std::u8streampos>::value);
static_assert(std::is_same<std::char_traits<char8_t>::state_type, std::mbstate_t>::value);
static_assert(std::is_same_v<std::char_traits<char8_t>::comparison_category, std::strong_ordering>);
#endif

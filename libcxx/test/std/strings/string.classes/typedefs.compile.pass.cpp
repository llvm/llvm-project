//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Test for the existence of:

// basic_string typedef names
// typedef basic_string<char>     string;
// typedef basic_string<char16_t> u16string;
// typedef basic_string<char8_t>  u8string;  // C++20
// typedef basic_string<char32_t> u32string;
// typedef basic_string<wchar_t>  wstring;

#include <string>
#include <type_traits>

#include "test_macros.h"

static_assert((std::is_same<std::string, std::basic_string<char> >::value), "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert((std::is_same<std::wstring, std::basic_string<wchar_t> >::value), "");
#endif
#ifndef TEST_HAS_NO_CHAR8_T
static_assert((std::is_same<std::u8string, std::basic_string<char8_t> >::value), "");
#endif
static_assert((std::is_same<std::u16string, std::basic_string<char16_t> >::value), "");
static_assert((std::is_same<std::u32string, std::basic_string<char32_t> >::value), "");

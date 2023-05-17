//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <string>
#include <utility>

#include "test_macros.h"

static_assert(!noexcept(std::operator""s(std::declval<const char*>(), std::declval<int>())), "");
#ifndef TEST_HAS_NO_CHAR8_T
static_assert(!noexcept(std::operator""s(std::declval<const char8_t*>(), std::declval<int>())), "");
#endif
static_assert(!noexcept(std::operator""s(std::declval<const char16_t*>(), std::declval<int>())), "");
static_assert(!noexcept(std::operator""s(std::declval<const char32_t*>(), std::declval<int>())), "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!noexcept(std::operator""s(std::declval<const wchar_t*>(), std::declval<int>())), "");
#endif

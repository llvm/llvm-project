//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// UNSUPPORTED: no-wide-characters

// Check that __constexpr_* cwchar functions are actually constexpr

#include <cwchar>

static_assert(std::__constexpr_wcslen(L"Banane") == 6, "");
static_assert(std::__constexpr_wmemcmp(L"Banane", L"Banand", 6) == 1, "");
static_assert(std::__constexpr_wmemcmp(L"Banane", L"Banane", 6) == 0, "");
static_assert(std::__constexpr_wmemcmp(L"Banane", L"Bananf", 6) == -1, "");

constexpr bool test_constexpr_wmemchr() {
  const wchar_t str[] = L"Banane";
  return std::__constexpr_wmemchr(str, 'n', 6) == str + 2;
}
static_assert(test_constexpr_wmemchr(), "");

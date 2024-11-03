//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

// Check that __constexpr_* cstring functions are actually constexpr

#include <__string/constexpr_c_functions.h>

static_assert(std::__constexpr_strlen("Banane") == 6, "");
static_assert(std::__constexpr_memcmp("Banane", "Banand", 6) == 1, "");
static_assert(std::__constexpr_memcmp("Banane", "Banane", 6) == 0, "");
static_assert(std::__constexpr_memcmp("Banane", "Bananf", 6) == -1, "");

constexpr bool test_constexpr_wmemchr() {
  const char str[] = "Banane";
  return std::__constexpr_char_memchr(str, 'n', 6) == str + 2;
}
static_assert(test_constexpr_wmemchr(), "");

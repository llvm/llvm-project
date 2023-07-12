//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Check that __constexpr_* cstring functions are actually constexpr

#include <__string/constexpr_c_functions.h>

constexpr unsigned char Banand[] = "Banand";
constexpr unsigned char Banane[] = "Banane";
constexpr unsigned char Bananf[] = "Bananf";

static_assert(std::__constexpr_strlen("Banane") == 6, "");
static_assert(std::__constexpr_memcmp(Banane, Banand, std::__element_count(6)) == 1, "");
static_assert(std::__constexpr_memcmp(Banane, Banane, std::__element_count(6)) == 0, "");
static_assert(std::__constexpr_memcmp(Banane, Bananf, std::__element_count(6)) == -1, "");

static_assert(!std::__constexpr_memcmp_equal(Banane, Banand, std::__element_count(6)), "");
static_assert(std::__constexpr_memcmp_equal(Banane, Banane, std::__element_count(6)), "");


constexpr bool test_constexpr_wmemchr() {
  const char str[] = "Banane";
  return std::__constexpr_memchr(str, 'n', 6) == str + 2;
}
static_assert(test_constexpr_wmemchr(), "");

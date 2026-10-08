//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// UNSUPPORTED: c++03

// <locale>

// void swap(locale &, locale &) noexcept

#include <cassert>
#include <locale>

#include "platform_support.h"

int main(int, char**) {
  std::locale loc1 = std::locale::classic();
  std::locale loc2(LOCALE_en_US_UTF_8);

  assert(loc1 != loc2);

  std::locale expected_lhs = loc2;
  std::locale expected_rhs = loc1;

  static_assert(noexcept(swap(loc1, loc2)), "");

  swap(loc1, loc2);
  assert(loc1 == expected_lhs);
  assert(loc2 == expected_rhs);

  return 0;
}

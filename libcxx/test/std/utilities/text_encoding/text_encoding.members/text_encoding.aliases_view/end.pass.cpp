//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// struct text_encoding::aliases_view

// Concerns:
// 1. begin() of an aliases_view() of "other" is equal to end()
// 2. That pushing an iterator beyond its range will result in it comparing equal to its end()

#include "test_text_encoding.h"

int main() {
  auto te1      = std::text_encoding("US-ASCII");
  auto te2      = std::text_encoding("UTF_8");
  auto other_te = std::text_encoding(std::text_encoding::other);
  auto a_1 = te1.aliases(), a_2 = te2.aliases();
  auto a_other = other_te.aliases();

  auto iter1 = a_1.begin(), iter2 = a_2.begin();
  // 1.
  assert(a_other.begin() == a_other.end());

  // 2.
  assert((iter1 + 1000) == std::ranges::end(a_1));
  assert((iter1 + 1000) == a_1.end());
  assert((iter2 + 1000) == std::ranges::end(a_2));
  assert((iter2 + 1000) == a_2.end());
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::aliases_view::end()

#include <cassert>
#include <text_encoding>

constexpr bool test() {
  std::text_encoding a_other = std::text_encoding("foobar");

  std::text_encoding::aliases_view other_aliases = a_other.aliases();

  // 1. begin() of an aliases_view of "other" is equal to end()
  assert(other_aliases.begin() == other_aliases.end());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

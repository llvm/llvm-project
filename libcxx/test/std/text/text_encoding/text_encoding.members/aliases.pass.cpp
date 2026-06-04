//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// constexpr text_encoding::aliases()

#include <cassert>
#include <string_view>
#include <text_encoding>

#include "test_macros.h"

constexpr bool test() {
  static_assert(noexcept(std::text_encoding().aliases()));
  ASSERT_SAME_TYPE(decltype(std::text_encoding().aliases()), std::text_encoding::aliases_view);

  // 2 aliases
  std::text_encoding utf8 = std::text_encoding::UTF8;

  auto aliases = utf8.aliases();

  assert(aliases.size() == 2);
  assert(std::string_view(aliases[0]) == "UTF-8");
  assert(std::string_view(aliases[1]) == "csUTF8");

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

// <text_encoding>

// text_encoding::aliases_view::operator[]

#include <cassert>
#include <string_view>
#include <text_encoding>

#include "test_macros.h"

constexpr bool test() {
  std::text_encoding te(std::text_encoding::id::UTF8);
  std::text_encoding::aliases_view aliases = te.aliases();
  auto iter                                = aliases.begin();

  ASSERT_SAME_TYPE(decltype(aliases[0]), const char*);
  assert(std::string_view(aliases[0]) == *iter);
  assert(std::string_view(aliases[1]) == std::string_view(*(iter + 1)));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}

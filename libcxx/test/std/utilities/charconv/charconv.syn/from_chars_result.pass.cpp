//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <charconv>

// struct from_chars_result
//   friend bool operator==(const from_chars_result&, const from_chars_result&) = default;

#include <charconv>

#include <cassert>
#include <compare>
#include <concepts>

#include "test_macros.h"

constexpr bool test() {
  std::from_chars_result lhs{nullptr, std::errc{}};
  std::from_chars_result rhs{nullptr, std::errc{}};
  assert(lhs == rhs);
  assert(!(lhs != rhs));

  return true;
}

int main(int, char**) {
  static_assert(std::equality_comparable<std::from_chars_result>);
  static_assert(!std::totally_ordered<std::from_chars_result>);
  static_assert(!std::three_way_comparable<std::from_chars_result>);

  assert(test());
  static_assert(test());

  return 0;
}

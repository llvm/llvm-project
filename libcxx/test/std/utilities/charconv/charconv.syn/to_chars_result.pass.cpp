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

// struct to_chars_result
//   friend bool operator==(const to_chars_result&, const to_chars_result&) = default;

#include <charconv>

#include <cassert>
#include <compare>
#include <concepts>

#include "test_macros.h"

constexpr bool test() {
  std::to_chars_result lhs{nullptr, std::errc{}};
  std::to_chars_result rhs{nullptr, std::errc{}};
  assert(lhs == rhs);
  assert(!(lhs != rhs));

  return true;
}

int main(int, char**) {
  static_assert(std::equality_comparable<std::to_chars_result>);
  static_assert(!std::totally_ordered<std::to_chars_result>);
  static_assert(!std::three_way_comparable<std::to_chars_result>);

  assert(test());
  static_assert(test());

  return 0;
}

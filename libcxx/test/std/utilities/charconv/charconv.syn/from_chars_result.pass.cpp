//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: c++20 && libcpp-no-concepts

// <charconv>

// struct from_chars_result
//   friend bool operator==(const from_chars_result&, const from_chars_result&) = default;

// [charconv.syn]/2
// The types to_chars_result and from_chars_result have the data members and
// special members specified above. They have no base classes or members other
// than those specified.

#include <charconv>

#include <cassert>
#include <compare>
#include <concepts>

#include "test_macros.h"

constexpr bool test() {
  std::from_chars_result lhs{nullptr, std::errc{}};
#if TEST_STD_VER > 17
  std::from_chars_result rhs{nullptr, std::errc{}};
  assert(lhs == rhs);
  assert(!(lhs != rhs));
#endif
  auto [ptr, ec] = lhs;
  static_assert(std::is_same_v<decltype(ptr), const char*>);
  assert(ptr == lhs.ptr);
  static_assert(std::is_same_v<decltype(ec), std::errc>);
  assert(ec == lhs.ec);

  return true;
}

int main(int, char**) {
#if TEST_STD_VER > 17
  static_assert(std::equality_comparable<std::from_chars_result>);
  static_assert(!std::totally_ordered<std::from_chars_result>);
  static_assert(!std::three_way_comparable<std::from_chars_result>);
#endif

  assert(test());
  static_assert(test());

  return 0;
}

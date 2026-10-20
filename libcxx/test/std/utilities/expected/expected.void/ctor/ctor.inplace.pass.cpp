//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit expected(in_place_t) noexcept;

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

// test explicit
static_assert(std::is_constructible_v<std::expected<void, int>, std::in_place_t>);
static_assert(!std::is_convertible_v<std::in_place_t, std::expected<void, int>>);

// test noexcept
static_assert(std::is_nothrow_constructible_v<std::expected<void, int>, std::in_place_t>);

constexpr bool test() {
  std::expected<void, int> e(std::in_place);
  assert(e.has_value());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

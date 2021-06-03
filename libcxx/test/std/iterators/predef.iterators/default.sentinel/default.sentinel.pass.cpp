//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// struct default_sentinel_t;
// inline constexpr default_sentinel_t default_sentinel;

#include <iterator>

#include <concepts>
#include <type_traits>

#include "test_macros.h"

int main(int, char**) {
  static_assert(std::is_empty_v<std::default_sentinel_t>);
  static_assert(std::semiregular<std::default_sentinel_t>);

  static_assert(std::same_as<decltype(std::default_sentinel), const std::default_sentinel_t>);

  std::default_sentinel_t s1;
  auto s2 = std::default_sentinel_t{};
  s2 = s1;

  return 0;
}

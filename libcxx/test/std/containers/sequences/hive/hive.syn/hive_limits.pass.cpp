//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// struct hive_limits {
//   size_t min;
//   size_t max;
//   constexpr hive_limits(size_t minimum, size_t maximum) noexcept
//     : min(minimum), max(maximum) {}
// };

#include <cassert>
#include <cstddef>
#include <hive>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_same_v<decltype(std::hive_limits::min), std::size_t>);
static_assert(std::is_same_v<decltype(std::hive_limits::max), std::size_t>);
static_assert(std::is_nothrow_constructible_v<std::hive_limits, std::size_t, std::size_t>);
static_assert(!std::is_default_constructible_v<std::hive_limits>);
static_assert(std::is_trivially_copyable_v<std::hive_limits>);

constexpr bool test() {
  std::hive_limits l(3, 42);
  assert(l.min == 3);
  assert(l.max == 42);

  l.min = 7;
  assert(l.min == 7);

  std::hive_limits copy = l;
  assert(copy.min == 7);
  assert(copy.max == 42);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

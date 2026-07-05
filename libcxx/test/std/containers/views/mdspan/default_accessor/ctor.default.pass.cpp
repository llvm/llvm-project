//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// Test default construction:
//
// constexpr default_accessor() noexcept = default;

#include <mdspan>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T>
constexpr void test_construction() {
  ASSERT_NOEXCEPT(std::default_accessor<T>{});
  [[maybe_unused]] std::default_accessor<T> acc;
  static_assert(std::is_trivially_default_constructible_v<std::default_accessor<T>>);
}

constexpr bool test() {
  test_construction<int>();
  test_construction<const int>();
  test_construction<MinimalElementType>();
  test_construction<const MinimalElementType>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

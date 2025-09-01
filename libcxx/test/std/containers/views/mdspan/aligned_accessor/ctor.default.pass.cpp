//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// Test default construction:
//
// constexpr aligned_accessor() noexcept = default;

#include <mdspan>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T, std::size_t N>
constexpr void test_construction() {
  ASSERT_NOEXCEPT(std::aligned_accessor<T, N>{});
  [[maybe_unused]] std::aligned_accessor<T, N> acc;
  static_assert(std::is_trivially_default_constructible_v<std::aligned_accessor<T, N>>);
}

template <class T>
constexpr void test_it() {
  constexpr std::size_t N = alignof(T);
  test_construction<T, N>();
  test_construction<T, 2 * N>();
  test_construction<T, 4 * N>();
  test_construction<T, 8 * N>();
  test_construction<T, 16 * N>();
  test_construction<T, 32 * N>();
}

constexpr bool test() {
  test_it<int>();
  test_it<const int>();
  test_it<MinimalElementType>();
  test_it<const MinimalElementType>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// static void shrink_to_fit() noexcept;

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <typename T, std::size_t N>
constexpr void test() {
  using V = std::inplace_vector<T, N>;
  static_assert((V::shrink_to_fit(), true));
  ASSERT_NOEXCEPT(V::shrink_to_fit());
}

constexpr bool tests() {
  test<int, 10>();
  test<int, 0>();
  test<MoveOnly, 10>();
  test<MoveOnly, 0>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}

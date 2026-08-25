//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void swap(inplace_vector& x)
//   noexcept(N == 0 || (is_nothrow_swappable_v<T> && is_nothrow_move_constructible_v<T>));

#include <cassert>
#include <inplace_vector>
#include <utility>

#include "../common.h"
#include "test_macros.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) noexcept(false) {}
  ThrowingMove& operator=(ThrowingMove&&) noexcept(false) { return *this; }
};

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c1{1, 2, 3};
    std::inplace_vector<int, 8> c2{4, 5};
    c1.swap(c2);
    assert_inplace_vector_equal(c1, {4, 5});
    assert_inplace_vector_equal(c2, {1, 2, 3});
  }
  {
    std::inplace_vector<int, 8> c1{1, 2, 3};
    std::inplace_vector<int, 8> c2{4, 5};
    swap(c1, c2);
    assert_inplace_vector_equal(c1, {4, 5});
    assert_inplace_vector_equal(c2, {1, 2, 3});
  }

  { // noexcept(N == 0 || (is_nothrow_swappable_v<T> && is_nothrow_move_constructible_v<T>))
    using C0 = std::inplace_vector<ThrowingMove, 0>;
    using C8 = std::inplace_vector<ThrowingMove, 8>;
    std::inplace_vector<int, 8> c1;
    std::inplace_vector<int, 8> c2;
    ASSERT_NOEXCEPT(swap(c1, c2));
    ASSERT_NOEXCEPT(swap(std::declval<C0&>(), std::declval<C0&>()));
    ASSERT_NOT_NOEXCEPT(swap(std::declval<C8&>(), std::declval<C8&>()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

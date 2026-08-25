//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector(inplace_vector&&)
//   noexcept(N == 0 || is_nothrow_move_constructible_v<T>);

#include <cassert>
#include <inplace_vector>
#include <type_traits>
#include <utility>

#include "../common.h"
#include "MoveOnly.h"
#include "test_macros.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) noexcept(false) {}
};

struct ThrowingCopyNothrowMove {
  ThrowingCopyNothrowMove() = default;
  ThrowingCopyNothrowMove(const ThrowingCopyNothrowMove&) noexcept(false) {}
  ThrowingCopyNothrowMove(ThrowingCopyNothrowMove&&) noexcept {}
};

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c{1, 2, 3};
    std::inplace_vector<int, 8> moved(std::move(c));
    assert_inplace_vector_equal(moved, {1, 2, 3});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<MoveOnly, 8> c;
    c.emplace_back(1);
    c.emplace_back(2);
    std::inplace_vector<MoveOnly, 8> moved(std::move(c));
    assert(moved.size() == 2);
    assert(moved[0].get() == 1);
    assert(moved[1].get() == 2);
  }

  { // noexcept(N == 0 || is_nothrow_move_constructible_v<T>)
    ASSERT_NOEXCEPT(std::inplace_vector<int, 8>(std::inplace_vector<int, 8>()));
    ASSERT_NOEXCEPT(std::inplace_vector<ThrowingMove, 0>(std::inplace_vector<ThrowingMove, 0>()));
    ASSERT_NOT_NOEXCEPT(std::inplace_vector<ThrowingMove, 8>(std::move(std::inplace_vector<ThrowingMove, 8>())));
    static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<ThrowingCopyNothrowMove, 8>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

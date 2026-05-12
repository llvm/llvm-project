//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector& operator=(inplace_vector&& other)
//   noexcept(N == 0 || (is_nothrow_move_assignable_v<T> && is_nothrow_move_constructible_v<T>));

#include <cassert>
#include <inplace_vector>
#include <type_traits>
#include <utility>

#include "../common.h"
#include "MoveOnly.h"
#include "test_macros.h"

struct ThrowingMoveAssign {
  ThrowingMoveAssign() = default;
  ThrowingMoveAssign(ThrowingMoveAssign&&) noexcept(false) {}
  ThrowingMoveAssign& operator=(ThrowingMoveAssign&&) noexcept(false) { return *this; }
};

struct ThrowingMoveCtorOnly {
  ThrowingMoveCtorOnly() = default;
  ThrowingMoveCtorOnly(ThrowingMoveCtorOnly&&) noexcept(false) {}
  ThrowingMoveCtorOnly& operator=(ThrowingMoveCtorOnly&&) noexcept { return *this; }
};

struct ThrowingMoveAssignOnly {
  ThrowingMoveAssignOnly() = default;
  ThrowingMoveAssignOnly(ThrowingMoveAssignOnly&&) noexcept = default;
  ThrowingMoveAssignOnly& operator=(ThrowingMoveAssignOnly&&) noexcept(false) { return *this; }
};

constexpr bool test() {
  {
    using C = std::inplace_vector<int, 8>;
    C c{1, 2, 3};
    C other{4, 5};
    ASSERT_SAME_TYPE(C&, decltype(c = std::move(other)));
    C& result = (c = std::move(other));
    assert(&result == &c);
    assert_inplace_vector_equal(c, {4, 5});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    using C = std::inplace_vector<MoveOnly, 8>;
    C c;
    c.emplace_back(1);
    C other;
    other.emplace_back(4);
    other.emplace_back(5);
    c = std::move(other);
    assert(c.size() == 2);
    assert(c[0].get() == 4);
    assert(c[1].get() == 5);
  }

  { // self-move-assignment leaves the container in a valid state
    std::inplace_vector<int, 8> c{1, 2, 3};
    std::inplace_vector<int, 8>& ref = c;
    c = std::move(ref);
    LIBCPP_ASSERT(c == (std::inplace_vector<int, 8>{1, 2, 3}));
  }

  { // noexcept(N == 0 || (is_nothrow_move_assignable_v<T> && is_nothrow_move_constructible_v<T>))
    ASSERT_NOEXCEPT(std::declval<std::inplace_vector<int, 8>&>() = std::declval<std::inplace_vector<int, 8>&&>());
    ASSERT_NOEXCEPT(std::declval<std::inplace_vector<ThrowingMoveAssign, 0>&>() =
                        std::declval<std::inplace_vector<ThrowingMoveAssign, 0>&&>());
    ASSERT_NOT_NOEXCEPT(std::declval<std::inplace_vector<ThrowingMoveAssign, 8>&>() =
                            std::declval<std::inplace_vector<ThrowingMoveAssign, 8>&&>());

    // both the move constructor and the move assignment of T must be noexcept
    static_assert(!std::is_nothrow_move_assignable_v<std::inplace_vector<ThrowingMoveCtorOnly, 8>>);
    static_assert(!std::is_nothrow_move_assignable_v<std::inplace_vector<ThrowingMoveAssignOnly, 8>>);
    static_assert(std::is_nothrow_move_assignable_v<std::inplace_vector<ThrowingMoveCtorOnly, 0>>);
    static_assert(std::is_nothrow_move_assignable_v<std::inplace_vector<ThrowingMoveAssignOnly, 0>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

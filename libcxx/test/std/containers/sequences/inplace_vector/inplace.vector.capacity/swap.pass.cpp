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

struct ThrowMove {
  constexpr ThrowMove()                            = default;
  constexpr ThrowMove(const ThrowMove&)            = default;
  constexpr ThrowMove& operator=(const ThrowMove&) = default;
  constexpr ThrowMove(ThrowMove&&) {}
  constexpr ThrowMove& operator=(ThrowMove&&) = default;
  constexpr ~ThrowMove()                      = default;
};

namespace MyNS {
struct M {
  M(M const&)            = delete;
  M& operator=(M const&) = delete;
};

void swap(M&&, M&&) noexcept {}

} // namespace MyNS

constexpr bool test() {
  std::inplace_vector<int, 8> c1{1, 2, 3};
  std::inplace_vector<int, 8> c2{4, 5};
  ASSERT_NOEXCEPT(c1.swap(c2));
  c1.swap(c2);
  assert(c1.capacity() == 8);
  assert(c2.capacity() == 8);
  assert_inplace_vector_equal(c1, {4, 5});
  assert_inplace_vector_equal(c2, {1, 2, 3});

  using namespace MyNS;
  {
    using C = std::inplace_vector<ThrowMove, 2>;
    ASSERT_NOT_NOEXCEPT(std::declval<C&>().swap(std::declval<C&>()));
  }

  {
    using C = std::inplace_vector<ThrowMove, 0>;
    ASSERT_NOEXCEPT(std::declval<C&>().swap(std::declval<C&>()));
  }

  {
    using C = std::inplace_vector<M, 4>;
    ASSERT_NOT_NOEXCEPT(std::declval<C&>().swap(std::declval<C&>()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

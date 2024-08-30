//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector(inplace_vector&&)
//     noexcept(N == 0 || is_nothrow_move_assignable_v<T>);

#include <inplace_vector>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <bool ConstructNoexcept, bool AssignNoexcept>
struct MoveOnly {
  MoveOnly(MoveOnly&&) noexcept(ConstructNoexcept);
  MoveOnly& operator=(MoveOnly&&) noexcept(AssignNoexcept);
  ~MoveOnly();
};

struct Immovable {
  Immovable(Immovable&&)            = delete;
  Immovable& operator=(Immovable&&) = delete;
  ~Immovable();
};

int main(int, char**) {
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<int, 0>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<int, 10>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<true, true>, 0>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<true, true>, 10>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<true, false>, 0>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<true, false>, 10>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<false, true>, 0>>);
  static_assert(!std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<false, true>, 10>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<false, false>, 0>>);
  static_assert(!std::is_nothrow_move_constructible_v<std::inplace_vector<MoveOnly<false, false>, 10>>);
  static_assert(std::is_nothrow_move_constructible_v<std::inplace_vector<Immovable, 0>>);
  static_assert(!std::is_nothrow_move_constructible_v<std::inplace_vector<Immovable, 10>>);

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(flat_multimap&&)
//        noexcept(is_nothrow_move_constructible<key_container_type>::value &&
//                 is_nothrow_move_constructible<mapped_container_type>::value &&
//                 is_nothrow_copy_constructible<key_compare>::value);

// This tests a conforming extension

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

struct ThrowingMoveComp {
  ThrowingMoveComp() = default;
  ThrowingMoveComp(const ThrowingMoveComp&) noexcept(true) {}
  ThrowingMoveComp(ThrowingMoveComp&&) noexcept(false) {}
  bool operator()(const auto&, const auto&) const { return false; }
};

struct MoveSensitiveComp {
  MoveSensitiveComp() noexcept(false)                  = default;
  MoveSensitiveComp(const MoveSensitiveComp&) noexcept = default;
  MoveSensitiveComp(MoveSensitiveComp&& rhs) { rhs.is_moved_from_ = true; }
  MoveSensitiveComp& operator=(const MoveSensitiveComp&) noexcept(false) = default;
  MoveSensitiveComp& operator=(MoveSensitiveComp&& rhs) {
    rhs.is_moved_from_ = true;
    return *this;
  }
  bool operator()(const auto&, const auto&) const { return false; }
  bool is_moved_from_ = false;
};

int main(int, char**) {
  {
    using C = std::flat_multimap<int, int>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  {
    using C = std::flat_multimap<int, int, std::less<int>, std::deque<int, test_allocator<int>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  {
    // Comparator fails to be nothrow-move-constructible
    using C = std::flat_multimap<int, int, ThrowingMoveComp>;
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  return 0;
}

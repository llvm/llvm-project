//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map& operator=(flat_map&& c)
//     noexcept(
//          is_nothrow_move_assignable<key_container_type>::value &&
//          is_nothrow_move_assignable<mapped_container_type>::value &&
//          is_nothrow_copy_assignable<key_compare>::value);

// This tests a conforming extension

#include <flat_map>
#include <functional>
#include <memory_resource>
#include <type_traits>
#include <vector>

#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_macros.h"

struct MoveSensitiveComp {
  MoveSensitiveComp() noexcept(false)                         = default;
  MoveSensitiveComp(const MoveSensitiveComp&) noexcept(false) = default;
  MoveSensitiveComp(MoveSensitiveComp&& rhs) { rhs.is_moved_from_ = true; }
  MoveSensitiveComp& operator=(const MoveSensitiveComp&) noexcept = default;
  MoveSensitiveComp& operator=(MoveSensitiveComp&& rhs) {
    rhs.is_moved_from_ = true;
    return *this;
  }
  bool operator()(const auto&, const auto&) const { return false; }
  bool is_moved_from_ = false;
};

struct MoveThrowsComp {
  MoveThrowsComp(MoveThrowsComp&&) noexcept(false);
  MoveThrowsComp(const MoveThrowsComp&) noexcept(true);
  MoveThrowsComp& operator=(MoveThrowsComp&&) noexcept(false);
  MoveThrowsComp& operator=(const MoveThrowsComp&) noexcept(true);
  bool operator()(const auto&, const auto&) const;
};

int main(int, char**) {
  {
    using C = std::flat_map<int, int>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C =
        std::flat_map<MoveOnly,
                      int,
                      std::less<MoveOnly>,
                      std::vector<MoveOnly, test_allocator<MoveOnly>>,
                      std::vector<int, test_allocator<int>>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C =
        std::flat_map<int,
                      MoveOnly,
                      std::less<int>,
                      std::vector<int, test_allocator<int>>,
                      std::vector<MoveOnly, test_allocator<MoveOnly>>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C =
        std::flat_map<MoveOnly,
                      int,
                      std::less<MoveOnly>,
                      std::vector<MoveOnly, other_allocator<MoveOnly>>,
                      std::vector<int, other_allocator<int>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C =
        std::flat_map<int,
                      MoveOnly,
                      std::less<int>,
                      std::vector<int, other_allocator<int>>,
                      std::vector<MoveOnly, other_allocator<MoveOnly>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    // Test with a comparator that throws on move-assignment.
    using C = std::flat_map<int, int, MoveThrowsComp>;
    LIBCPP_STATIC_ASSERT(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    // Test with a container that throws on move-assignment.
    using C = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::vector<int>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    // Test with a container that throws on move-assignment.
    using C = std::flat_map<int, int, std::less<int>, std::vector<int>, std::pmr::vector<int>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }

  return 0;
}

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

template <class T>
struct ThrowingMoveAllocator {
  using value_type                                    = T;
  explicit ThrowingMoveAllocator()                    = default;
  ThrowingMoveAllocator(const ThrowingMoveAllocator&) = default;
  ThrowingMoveAllocator(ThrowingMoveAllocator&&) noexcept(false) {}
  T* allocate(std::ptrdiff_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::ptrdiff_t n) { return std::allocator<T>().deallocate(p, n); }
  friend bool operator==(ThrowingMoveAllocator, ThrowingMoveAllocator) = default;
};

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
#if _LIBCPP_VERSION
  {
    // Container fails to be nothrow-move-constructible; this relies on libc++'s support for non-nothrow-copyable allocators
    using C =
        std::flat_multimap<int, int, std::less<int>, std::deque<int, ThrowingMoveAllocator<int>>, std::vector<int>>;
    static_assert(!std::is_nothrow_move_constructible_v<std::deque<int, ThrowingMoveAllocator<int>>>);
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  {
    // Container fails to be nothrow-move-constructible; this relies on libc++'s support for non-nothrow-copyable allocators
    using C =
        std::flat_multimap<int, int, std::less<int>, std::vector<int>, std::deque<int, ThrowingMoveAllocator<int>>>;
    static_assert(!std::is_nothrow_move_constructible_v<std::deque<int, ThrowingMoveAllocator<int>>>);
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
#endif // _LIBCPP_VERSION
  {
    // Comparator fails to be nothrow-move-constructible
    using C = std::flat_multimap<int, int, ThrowingMoveComp>;
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  return 0;
}

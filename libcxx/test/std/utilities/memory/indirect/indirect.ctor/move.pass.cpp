//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// constexpr indirect(indirect&& other) noexcept;

// constexpr indirect(allocator_arg_t, const Allocator& a, indirect&& other)
//   noexcept(allocator_traits<Allocator>::is_always_equal::value);

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_allocator.h"
#include "test_convertible.h"

constexpr void test_ctor_not_explicit() {
  static_assert(test_convertible<std::indirect<int>, std::indirect<int>&&>());
  static_assert(
      test_convertible<std::indirect<int>, std::allocator_arg_t, const std::allocator<int>&, std::indirect<int>&&>());
}

constexpr void test_ctor_noexcept() {
  static_assert(std::is_nothrow_constructible_v<std::indirect<int, test_allocator<int>>,
                                                std::indirect<int, test_allocator<int>>&&>);
  static_assert(std::is_nothrow_constructible_v<std::indirect<int>,
                                                std::allocator_arg_t,
                                                const std::allocator<int>&,
                                                std::indirect<int>&&>);
}

constexpr void test_ctor() {
  {
    std::indirect<int> i1(42);

    // Moving from an indirect leaves it valueless.
    std::indirect<int> i2(std::move(i1));
    assert(i1.valueless_after_move());
    assert(!i2.valueless_after_move());
    assert(*i2 == 42);

    // Move constructing from a valueless indirect creates a new valueless indirect.
    std::indirect<int> i3(std::move(i1));
    assert(i1.valueless_after_move());
    assert(i3.valueless_after_move());
  }
  {
    test_allocator_statistics stats;
    std::indirect<int, test_allocator<int>> i1(10);

    // If the allocators are equal, no memory is allocated.
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(&stats), std::move(i1));
    assert(i1.valueless_after_move());
    assert(!i2.valueless_after_move());
    assert(*i2 == 10);
    assert(stats.construct_count == 0);
  }
  {
    test_allocator_statistics stats;
    std::indirect<int, test_allocator<int>> i1(10);

    // If the allocators aren't equal, a new owned object is constructed.
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(42, &stats), std::move(i1));
    assert(i1.valueless_after_move());
    assert(!i2.valueless_after_move());
    assert(*i2 == 10);
    assert(i1.get_allocator().get_data() == 0);
    assert(i2.get_allocator().get_data() == 42);
    assert(stats.construct_count == 1);

    // If the source object is valueless, no memory is allocated, even if the allocators aren't equal.
    std::indirect<int, test_allocator<int>> i3(std::allocator_arg, test_allocator<int>(67, &stats), std::move(i1));
    assert(i1.valueless_after_move());
    assert(i3.valueless_after_move());
    assert(i1.get_allocator().get_data() == 0);
    assert(i3.get_allocator().get_data() == 67);
    assert(stats.construct_count == 1);
  }
// Temporary hack. Will need to fix before merging.
#if 0
  struct Incomplete;
  { // Move construction doesn't require T to be complete.
    (void)([](std::indirect<Incomplete>&& i) -> std::indirect<Incomplete> { return {std::move(i)}; });
  }
  { // Uses-allocator move construction doesn't require T to be complete as long as the allocator is always equal.
    (void)([](std::indirect<Incomplete>&& i) -> std::indirect<Incomplete> {
      return {std::allocator_arg, std::allocator<Incomplete>(), std::move(i)};
    });
  }
#endif
}

constexpr bool test() {
  test_ctor_not_explicit();
  test_ctor_noexcept();
  test_ctor();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

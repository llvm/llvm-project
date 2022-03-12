//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// owning_view() requires default_initializable<R> = default;
// constexpr owning_view(R&& t);

#include <ranges>

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct DefaultConstructible {
  int i;
  constexpr explicit DefaultConstructible(int j = 42) : i(j) {}
  int *begin() const;
  int *end() const;
};

struct NotDefaultConstructible {
  int i;
  constexpr explicit NotDefaultConstructible(int j) : i(j) {}
  int *begin() const;
  int *end() const;
};

struct MoveChecker {
  int i;
  constexpr explicit MoveChecker(int j) : i(j) {}
  constexpr MoveChecker(MoveChecker&& v) : i(std::exchange(v.i, -1)) {}
  MoveChecker& operator=(MoveChecker&&);
  int *begin() const;
  int *end() const;
};

struct NoexceptChecker {
  int *begin() const;
  int *end() const;
};

constexpr bool test()
{
  {
    using OwningView = std::ranges::owning_view<DefaultConstructible>;
    static_assert(std::is_constructible_v<OwningView>);
    static_assert(std::default_initializable<OwningView>);
    static_assert(std::movable<OwningView>);
    static_assert(std::is_trivially_move_constructible_v<OwningView>);
    static_assert(std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!std::is_copy_constructible_v<OwningView>);
    static_assert(!std::is_copy_assignable_v<OwningView>);
    static_assert(!std::is_constructible_v<OwningView, int>);
    static_assert(!std::is_constructible_v<OwningView, DefaultConstructible&>);
    static_assert(std::is_constructible_v<OwningView, DefaultConstructible&&>);
    static_assert(!std::is_convertible_v<int, OwningView>);
    static_assert(std::is_convertible_v<DefaultConstructible&&, OwningView>);
    {
      OwningView ov;
      assert(ov.base().i == 42);
    }
    {
      OwningView ov = OwningView(DefaultConstructible(1));
      assert(ov.base().i == 1);
    }
  }
  {
    using OwningView = std::ranges::owning_view<NotDefaultConstructible>;
    static_assert(!std::is_constructible_v<OwningView>);
    static_assert(!std::default_initializable<OwningView>);
    static_assert(std::movable<OwningView>);
    static_assert(std::is_trivially_move_constructible_v<OwningView>);
    static_assert(std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!std::is_copy_constructible_v<OwningView>);
    static_assert(!std::is_copy_assignable_v<OwningView>);
    static_assert(!std::is_constructible_v<OwningView, int>);
    static_assert(!std::is_constructible_v<OwningView, NotDefaultConstructible&>);
    static_assert(std::is_constructible_v<OwningView, NotDefaultConstructible&&>);
    static_assert(!std::is_convertible_v<int, OwningView>);
    static_assert(std::is_convertible_v<NotDefaultConstructible&&, OwningView>);
    {
      OwningView ov = OwningView(NotDefaultConstructible(1));
      assert(ov.base().i == 1);
    }
  }
  {
    using OwningView = std::ranges::owning_view<MoveChecker>;
    static_assert(!std::is_constructible_v<OwningView>);
    static_assert(!std::default_initializable<OwningView>);
    static_assert(std::movable<OwningView>);
    static_assert(!std::is_trivially_move_constructible_v<OwningView>);
    static_assert(!std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!std::is_copy_constructible_v<OwningView>);
    static_assert(!std::is_copy_assignable_v<OwningView>);
    static_assert(!std::is_constructible_v<OwningView, int>);
    static_assert(!std::is_constructible_v<OwningView, MoveChecker&>);
    static_assert(std::is_constructible_v<OwningView, MoveChecker&&>);
    static_assert(!std::is_convertible_v<int, OwningView>);
    static_assert(std::is_convertible_v<MoveChecker&&, OwningView>);
    {
      // Check that the constructor does indeed move from the target object.
      auto m = MoveChecker(42);
      OwningView ov = OwningView(std::move(m));
      assert(ov.base().i == 42);
      assert(m.i == -1);
    }
  }
  {
    // Check that the defaulted constructors are (not) noexcept when appropriate.

    static_assert( std::is_nothrow_constructible_v<NoexceptChecker>); // therefore,
    static_assert( std::is_nothrow_constructible_v<std::ranges::owning_view<NoexceptChecker>>);
    static_assert(!std::is_nothrow_constructible_v<DefaultConstructible>); // therefore,
    static_assert(!std::is_nothrow_constructible_v<std::ranges::owning_view<DefaultConstructible>>);

    static_assert( std::is_nothrow_move_constructible_v<NoexceptChecker>); // therefore,
    static_assert( std::is_nothrow_move_constructible_v<std::ranges::owning_view<NoexceptChecker>>);
    static_assert(!std::is_nothrow_move_constructible_v<MoveChecker>); // therefore,
    static_assert(!std::is_nothrow_move_constructible_v<std::ranges::owning_view<MoveChecker>>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

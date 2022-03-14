//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::views::transform

#include <ranges>

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "types.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct NonCopyableFunction {
  NonCopyableFunction(NonCopyableFunction const&) = delete;
  template <class T>
  constexpr T operator()(T x) const { return x; }
};

constexpr bool test() {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Test `views::transform(f)(v)`
  {
    {
      using Result = std::ranges::transform_view<MoveOnlyView, PlusOne>;
      std::same_as<Result> auto result = std::views::transform(PlusOne{})(MoveOnlyView{buff});
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
    {
      auto const partial = std::views::transform(PlusOne{});
      using Result = std::ranges::transform_view<MoveOnlyView, PlusOne>;
      std::same_as<Result> auto result = partial(MoveOnlyView{buff});
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
  }

  // Test `v | views::transform(f)`
  {
    {
      using Result = std::ranges::transform_view<MoveOnlyView, PlusOne>;
      std::same_as<Result> auto result = MoveOnlyView{buff} | std::views::transform(PlusOne{});
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
    {
      auto const partial = std::views::transform(PlusOne{});
      using Result = std::ranges::transform_view<MoveOnlyView, PlusOne>;
      std::same_as<Result> auto result = MoveOnlyView{buff} | partial;
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
  }

  // Test `views::transform(v, f)`
  {
    using Result = std::ranges::transform_view<MoveOnlyView, PlusOne>;
    std::same_as<Result> auto result = std::views::transform(MoveOnlyView{buff}, PlusOne{});
    assert(result.begin().base() == buff);
    assert(result[0] == 1);
    assert(result[1] == 2);
    assert(result[2] == 3);
  }

  // Test that one can call std::views::transform with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X { };
    auto partial = std::views::transform(X{});
    (void)partial;
  }

  // Test `adaptor | views::transform(f)`
  {
    {
      using Result = std::ranges::transform_view<std::ranges::transform_view<MoveOnlyView, PlusOne>, TimesTwo>;
      std::same_as<Result> auto result = MoveOnlyView{buff} | std::views::transform(PlusOne{}) | std::views::transform(TimesTwo{});
      assert(result.begin().base().base() == buff);
      assert(result[0] == 2);
      assert(result[1] == 4);
      assert(result[2] == 6);
    }
    {
      auto const partial = std::views::transform(PlusOne{}) | std::views::transform(TimesTwo{});
      using Result = std::ranges::transform_view<std::ranges::transform_view<MoveOnlyView, PlusOne>, TimesTwo>;
      std::same_as<Result> auto result = MoveOnlyView{buff} | partial;
      assert(result.begin().base().base() == buff);
      assert(result[0] == 2);
      assert(result[1] == 4);
      assert(result[2] == 6);
    }
  }

  // Test SFINAE friendliness
  {
    struct NotAView { };
    struct NotInvocable { };

    static_assert(!CanBePiped<MoveOnlyView, decltype(std::views::transform)>);
    static_assert( CanBePiped<MoveOnlyView, decltype(std::views::transform(PlusOne{}))>);
    static_assert(!CanBePiped<NotAView,       decltype(std::views::transform(PlusOne{}))>);
    static_assert(!CanBePiped<MoveOnlyView, decltype(std::views::transform(NotInvocable{}))>);

    static_assert(!std::is_invocable_v<decltype(std::views::transform)>);
    static_assert(!std::is_invocable_v<decltype(std::views::transform), PlusOne, MoveOnlyView>);
    static_assert( std::is_invocable_v<decltype(std::views::transform), MoveOnlyView, PlusOne>);
    static_assert(!std::is_invocable_v<decltype(std::views::transform), MoveOnlyView, PlusOne, PlusOne>);
    static_assert(!std::is_invocable_v<decltype(std::views::transform), NonCopyableFunction>);
  }

  {
    static_assert(std::is_same_v<decltype(std::ranges::views::transform), decltype(std::views::transform)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

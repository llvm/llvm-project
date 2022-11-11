//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::take_while

#include <algorithm>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "types.h"

struct Pred {
  constexpr bool operator()(int i) const { return i < 3; }
};

struct Foo {};

struct MoveOnlyView : IntBufferViewBase {
  using IntBufferViewBase::IntBufferViewBase;
  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&)                 = default;
  MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  constexpr const int* begin() const { return buffer_; }
  constexpr const int* end() const { return buffer_ + size_; }
};

static_assert(!std::is_invocable_v<decltype((std::views::take_while))>);
static_assert(std::is_invocable_v<decltype((std::views::take_while)), int>);
static_assert(std::is_invocable_v<decltype((std::views::take_while)), Pred>);
static_assert(!std::is_invocable_v<decltype((std::views::take_while)), int, Pred>);
static_assert(std::is_invocable_v<decltype((std::views::take_while)), int (&)[2], Pred>);
static_assert(!std::is_invocable_v<decltype((std::views::take_while)), Foo (&)[2], Pred>);
static_assert(std::is_invocable_v<decltype((std::views::take_while)), MoveOnlyView, Pred>);

template <class View, class T>
concept CanBePiped =
    requires(View&& view, T&& t) {
      { std::forward<View>(view) | std::forward<T>(t) };
    };

static_assert(!CanBePiped<MoveOnlyView, decltype(std::views::take_while)>);
static_assert(CanBePiped<MoveOnlyView, decltype(std::views::take_while(Pred{}))>);
static_assert(!CanBePiped<int, decltype(std::views::take_while(Pred{}))>);
static_assert(CanBePiped<int (&)[2], decltype(std::views::take_while(Pred{}))>);
static_assert(!CanBePiped<Foo (&)[2], decltype(std::views::take_while(Pred{}))>);

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 3, 2, 1};

  // Test `views::take_while(p)(v)`
  {
    using Result                               = std::ranges::take_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> decltype(auto) result = std::views::take_while(Pred{})(MoveOnlyView{buff});
    auto expected                              = {1, 2};
    assert(std::ranges::equal(result, expected));
  }
  {
    auto const partial                         = std::views::take_while(Pred{});
    using Result                               = std::ranges::take_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> decltype(auto) result = partial(MoveOnlyView{buff});
    auto expected                              = {1, 2};
    assert(std::ranges::equal(result, expected));
  }

  // Test `v | views::take_while(p)`
  {
    using Result                               = std::ranges::take_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> decltype(auto) result = MoveOnlyView{buff} | std::views::take_while(Pred{});
    auto expected                              = {1, 2};
    assert(std::ranges::equal(result, expected));
  }
  {
    auto const partial                         = std::views::take_while(Pred{});
    using Result                               = std::ranges::take_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> decltype(auto) result = MoveOnlyView{buff} | partial;
    auto expected                              = {1, 2};
    assert(std::ranges::equal(result, expected));
  }

  // Test `views::take_while(v, p)`
  {
    using Result                               = std::ranges::take_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> decltype(auto) result = std::views::take_while(MoveOnlyView{buff}, Pred{});
    auto expected                              = {1, 2};
    assert(std::ranges::equal(result, expected));
  }

  // Test adaptor | adaptor
  {
    struct Pred2 {
      constexpr bool operator()(int i) const { return i < 2; }
    };
    auto const partial = std::views::take_while(Pred{}) | std::views::take_while(Pred2{});
    using Result       = std::ranges::take_while_view<std::ranges::take_while_view<MoveOnlyView, Pred>, Pred2>;
    std::same_as<Result> decltype(auto) result = MoveOnlyView{buff} | partial;
    auto expected                              = {1};
    assert(std::ranges::equal(result, expected));
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

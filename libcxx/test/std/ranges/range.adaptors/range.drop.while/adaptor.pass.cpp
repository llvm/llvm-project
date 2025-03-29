//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::drop_while

#include <algorithm>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_range.h"

struct Pred {
  constexpr bool operator()(int i) const { return i < 3; }
};

struct Foo {};

template <class T>
struct BufferView : std::ranges::view_base {
  T* buffer_;
  std::size_t size_;

  template <std::size_t N>
  constexpr BufferView(T (&b)[N]) : buffer_(b), size_(N) {}
};

using IntBufferView = BufferView<int>;

struct MoveOnlyView : IntBufferView {
  using IntBufferView::IntBufferView;
  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&)                 = default;
  MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  constexpr const int* begin() const { return buffer_; }
  constexpr const int* end() const { return buffer_ + size_; }
};

static_assert(!std::is_invocable_v<decltype((std::views::drop_while))>);
static_assert(std::is_invocable_v<decltype((std::views::drop_while)), int>);
static_assert(std::is_invocable_v<decltype((std::views::drop_while)), Pred>);
static_assert(!std::is_invocable_v<decltype((std::views::drop_while)), int, Pred>);
static_assert(std::is_invocable_v<decltype((std::views::drop_while)), int (&)[2], Pred>);
static_assert(!std::is_invocable_v<decltype((std::views::drop_while)), Foo (&)[2], Pred>);
static_assert(std::is_invocable_v<decltype((std::views::drop_while)), MoveOnlyView, Pred>);

static_assert(!CanBePiped<MoveOnlyView, decltype(std::views::drop_while)>);
static_assert(CanBePiped<MoveOnlyView, decltype(std::views::drop_while(Pred{}))>);
static_assert(!CanBePiped<int, decltype(std::views::drop_while(Pred{}))>);
static_assert(CanBePiped<int (&)[2], decltype(std::views::drop_while(Pred{}))>);
static_assert(!CanBePiped<Foo (&)[2], decltype(std::views::drop_while(Pred{}))>);

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 3, 2, 1};

  // Test `views::drop_while(p)(v)`
  {
    using Result                     = std::ranges::drop_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> auto result = std::views::drop_while(Pred{})(MoveOnlyView{buff});
    auto expected                    = {3, 4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }
  {
    auto const partial               = std::views::drop_while(Pred{});
    using Result                     = std::ranges::drop_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> auto result = partial(MoveOnlyView{buff});
    auto expected                    = {3, 4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }

  // Test `v | views::drop_while(p)`
  {
    using Result                     = std::ranges::drop_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> auto result = MoveOnlyView{buff} | std::views::drop_while(Pred{});
    auto expected                    = {3, 4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }
  {
    auto const partial               = std::views::drop_while(Pred{});
    using Result                     = std::ranges::drop_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> auto result = MoveOnlyView{buff} | partial;
    auto expected                    = {3, 4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }

  // Test `views::drop_while(v, p)`
  {
    using Result                     = std::ranges::drop_while_view<MoveOnlyView, Pred>;
    std::same_as<Result> auto result = std::views::drop_while(MoveOnlyView{buff}, Pred{});
    auto expected                    = {3, 4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }

  // Test adaptor | adaptor
  {
    struct Pred2 {
      constexpr bool operator()(int i) const { return i < 4; }
    };
    auto const partial = std::views::drop_while(Pred{}) | std::views::drop_while(Pred2{});
    using Result       = std::ranges::drop_while_view<std::ranges::drop_while_view<MoveOnlyView, Pred>, Pred2>;
    std::same_as<Result> auto result = MoveOnlyView{buff} | partial;
    auto expected                    = {4, 3, 2, 1};
    assert(std::ranges::equal(result, expected));
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

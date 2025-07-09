//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

//     constexpr V base() const & requires copy_constructible<V> { return base_; }
//     constexpr V base() && { return std::move(base_); }

#include <cassert>
#include <concepts>
#include <ranges>

#include <utility>

#include "MoveOnly.h"

struct SimpleView : std::ranges::view_base {
  int i_;

  int* begin() const;
  int* end() const;
};

struct MoveOnlyView : public SimpleView {
  MoveOnly m_;
};

template <class T>
concept HasBase = requires(T&& t) { std::forward<T>(t).base(); };

static_assert(HasBase<std::ranges::to_input_view<SimpleView> const&>);
static_assert(HasBase<std::ranges::to_input_view<SimpleView>&&>);

static_assert(!HasBase<std::ranges::to_input_view<MoveOnlyView> const&>);
static_assert(HasBase<std::ranges::to_input_view<MoveOnlyView>&&>);

constexpr bool test() {
  { // &
    std::ranges::to_input_view<SimpleView> view(SimpleView{{}, 94});
    std::same_as<SimpleView> decltype(auto) v = view.base();
    assert(v.i_ == 94);
  }

  { // const &
    const std::ranges::to_input_view<SimpleView> view(SimpleView{{}, 94});
    std::same_as<SimpleView> decltype(auto) v = view.base();
    assert(v.i_ == 94);
  }

  { // &&
    std::ranges::to_input_view<SimpleView> view(SimpleView{{}, 94});
    std::same_as<SimpleView> decltype(auto) v = std::move(view).base();
    assert(v.i_ == 94);
  }

  { // const &&
    const std::ranges::to_input_view<SimpleView> view(SimpleView{{}, 94});
    std::same_as<SimpleView> decltype(auto) v = std::move(view).base();
    assert(v.i_ == 94);
  }

  { // move only
    std::ranges::to_input_view<MoveOnlyView> view(MoveOnlyView{{}, 94});
    std::same_as<MoveOnlyView> decltype(auto) v = std::move(view).base();
    assert(v.m_.get() == 94);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

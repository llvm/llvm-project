//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr V base() const & requires copy_constructible<V> { return base_; }
// constexpr V base() && { return std::move(base_); }

#include <cassert>
#include <ranges>
#include <utility>

#include "MoveOnly.h"

struct SimpleView : std::ranges::view_base {
  int i;
  int* begin() const;
  int* end() const;
};

struct MoveOnlyView : SimpleView {
  MoveOnly m;
};

template <class T>
concept HasBase = requires(T&& t) { std::forward<T>(t).base(); };

static_assert(HasBase<std::ranges::as_rvalue_view<SimpleView> const&>);
static_assert(HasBase<std::ranges::as_rvalue_view<SimpleView>&&>);

static_assert(!HasBase<std::ranges::as_rvalue_view<MoveOnlyView> const&>);
static_assert(HasBase<std::ranges::as_rvalue_view<MoveOnlyView>&&>);

constexpr bool test() {
  { // const &
    const std::ranges::as_rvalue_view<SimpleView> view(SimpleView{{}, 5});
    std::same_as<SimpleView> decltype(auto) v = view.base();
    assert(v.i == 5);
  }

  { // &
    std::ranges::as_rvalue_view<SimpleView> view(SimpleView{{}, 5});
    std::same_as<SimpleView> decltype(auto) v = view.base();
    assert(v.i == 5);
  }

  { // &&
    std::ranges::as_rvalue_view<SimpleView> view(SimpleView{{}, 5});
    std::same_as<SimpleView> decltype(auto) v = std::move(view).base();
    assert(v.i == 5);
  }

  { // const &&
    const std::ranges::as_rvalue_view<SimpleView> view(SimpleView{{}, 5});
    std::same_as<SimpleView> decltype(auto) v = std::move(view).base();
    assert(v.i == 5);
  }

  { // move only
    std::ranges::as_rvalue_view<MoveOnlyView> view(MoveOnlyView{{}, 5});
    std::same_as<MoveOnlyView> decltype(auto) v = std::move(view).base();
    assert(v.m.get() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

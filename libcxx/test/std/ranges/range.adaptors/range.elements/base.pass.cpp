//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr V base() const & requires copy_constructible<V> { return base_; }
// constexpr V base() && { return std::move(base_); }

#include <cassert>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"

struct View : std::ranges::view_base {
  int i;
  std::tuple<int>* begin() const;
  std::tuple<int>* end() const;
};

struct MoveOnlyView : View {
  MoveOnly mo;
};

template <class T>
concept HasBase = requires(T&& t) { std::forward<T>(t).base(); };

static_assert(HasBase<std::ranges::elements_view<View, 0> const&>);
static_assert(HasBase<std::ranges::elements_view<View, 0>&&>);

static_assert(!HasBase<std::ranges::elements_view<MoveOnlyView, 0> const&>);
static_assert(HasBase<std::ranges::elements_view<MoveOnlyView, 0>&&>);

constexpr bool test() {
  // const &
  {
    const std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    std::same_as<View> decltype(auto) v = ev.base();
    assert(v.i == 5);
  }

  // &
  {
    std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    std::same_as<View> decltype(auto) v = ev.base();
    assert(v.i == 5);
  }

  // &&
  {
    std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    std::same_as<View> decltype(auto) v = std::move(ev).base();
    assert(v.i == 5);
  }

  // const &&
  {
    const std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    std::same_as<View> decltype(auto) v = std::move(ev).base();
    assert(v.i == 5);
  }

  // move only
  {
    std::ranges::elements_view<MoveOnlyView, 0> ev{MoveOnlyView{{}, 5}};
    std::same_as<MoveOnlyView> decltype(auto) v = std::move(ev).base();
    assert(v.mo.get() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

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
#include <type_traits>
#include <utility>

#include "MoveOnly.h"

struct View : std::ranges::view_base {
  int i;
  int* begin() const;
  int* end() const;
};

struct MoveOnlyView : View {
  MoveOnly mo;
};

template <class T>
concept HasBase = requires(T&& t) { std::forward<T>(t).base(); };

static_assert(HasBase<std::ranges::split_view<View, View> const&>);
static_assert(HasBase<std::ranges::split_view<View, View>&&>);

static_assert(!HasBase<std::ranges::split_view<MoveOnlyView, View> const&>);
static_assert(HasBase<std::ranges::split_view<MoveOnlyView, View>&&>);

constexpr bool test() {
  // const &
  {
    const std::ranges::split_view<View, View> sv{View{{}, 5}, View{{}, 0}};
    std::same_as<View> decltype(auto) v = sv.base();
    assert(v.i == 5);
  }

  // &
  {
    std::ranges::split_view<View, View> sv{View{{}, 5}, View{{}, 0}};
    std::same_as<View> decltype(auto) v = sv.base();
    assert(v.i == 5);
  }

  // &&
  {
    std::ranges::split_view<View, View> sv{View{{}, 5}, View{{}, 0}};
    std::same_as<View> decltype(auto) v = std::move(sv).base();
    assert(v.i == 5);
  }

  // const &&
  {
    std::ranges::split_view<View, View> sv{View{{}, 5}, View{{}, 0}};
    std::same_as<View> decltype(auto) v = std::move(sv).base();
    assert(v.i == 5);
  }

  // move only
  {
    std::ranges::split_view<MoveOnlyView, View> sv{MoveOnlyView{{}, 5}, View{{}, 0}};
    std::same_as<MoveOnlyView> decltype(auto) v = std::move(sv).base();
    assert(v.mo.get() == 5);
  }

  // LWG 3590 split_view::base() const & is overconstrained
  {
    struct CopyCtorButNotAssignable : std::ranges::view_base {
      int i;
      constexpr CopyCtorButNotAssignable(int ii) : i(ii) {}
      constexpr CopyCtorButNotAssignable(const CopyCtorButNotAssignable&)            = default;
      constexpr CopyCtorButNotAssignable(CopyCtorButNotAssignable&&)                 = default;
      constexpr CopyCtorButNotAssignable& operator=(CopyCtorButNotAssignable&&)      = default;
      constexpr CopyCtorButNotAssignable& operator=(const CopyCtorButNotAssignable&) = delete;
      constexpr int* begin() const { return nullptr; }
      constexpr int* end() const { return nullptr; }
    };
    static_assert(std::copy_constructible<CopyCtorButNotAssignable>);
    static_assert(!std::copyable<CopyCtorButNotAssignable>);
    const std::ranges::split_view<CopyCtorButNotAssignable, CopyCtorButNotAssignable> sv{
        CopyCtorButNotAssignable{5}, CopyCtorButNotAssignable{5}};
    std::same_as<CopyCtorButNotAssignable> decltype(auto) v = sv.base();
    assert(v.i == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

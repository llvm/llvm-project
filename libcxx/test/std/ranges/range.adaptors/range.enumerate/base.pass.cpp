//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// constexpr V base() const & requires copy_constructible<V>;
// constexpr V base() &&;

#include <cassert>
#include <ranges>

#include "MoveOnly.h"

#include "types.h"

template <class T>
concept HasBase = requires(T&& t) { std::forward<T>(t).base(); };

// SFINAE

static_assert(HasBase<std::ranges::enumerate_view<RangeView> const&>);
static_assert(HasBase<std::ranges::enumerate_view<RangeView>&&>);

struct MoveOnlyView : RangeView {
  MoveOnly mo;
};

static_assert(!HasBase<std::ranges::enumerate_view<MoveOnlyView> const&>);
static_assert(HasBase<std::ranges::enumerate_view<MoveOnlyView>&&>);

constexpr bool test() {
  // Check the const& overload
  {
    int buff[] = {0, 1, 2, 3};

    RangeView range(buff, buff + 4);

    std::ranges::enumerate_view<RangeView> view{range};
    std::same_as<RangeView> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(range.begin() == result.begin());
    assert(range.end() == result.end());
  }
  {
    int buff[] = {0, 1, 2, 3};

    RangeView const range(buff, buff + 4);

    std::ranges::enumerate_view<RangeView> const view{range};
    std::same_as<RangeView> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(range.begin() == result.begin());
    assert(range.end() == result.end());
  }

  // Check the && overload
  {
    int buff[] = {0, 1, 2, 3};

    RangeView const range(buff, buff + 4);

    std::ranges::enumerate_view<RangeView> view{range};
    std::same_as<RangeView> decltype(auto) result = std::move(view).base();
    assert(result.wasMoveInitialized);
    assert(range.begin() == result.begin());
    assert(range.end() == result.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

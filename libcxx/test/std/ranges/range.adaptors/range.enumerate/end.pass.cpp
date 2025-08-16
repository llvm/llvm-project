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

// constexpr auto end() requires (!simple-view<V>);
// constexpr auto end() const requires range-with-movable-references<const V>;

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"

#include "types.h"

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of .end()
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    using Iterator = std::ranges::iterator_t<decltype(view)>;
    static_assert(std::same_as<Iterator, decltype(view.end())>);
    using Sentinel = std::ranges::sentinel_t<decltype(view)>;
    static_assert(std::same_as<Sentinel, decltype(view.end())>);
  }

  // Check the return type of .end() const
  {
    RangeView range(buff, buff + 1);

    const std::ranges::enumerate_view view(range);
    using Iterator = std::ranges::iterator_t<decltype(view)>;
    static_assert(std::same_as<Iterator, decltype(view.end())>);
    using Sentinel = std::ranges::sentinel_t<decltype(view)>;
    static_assert(std::same_as<Sentinel, decltype(view.end())>);
  }

  // end() over an empty range
  {
    RangeView range(buff, buff);

    std::ranges::enumerate_view view(range);

    auto it = view.end();
    assert(base(it.base()) == buff);
    assert(it == view.end());

    auto constIt = std::as_const(view).end();
    assert(base(constIt.base()) == buff);
    assert(constIt == std::as_const(view).end());
  }

  // end() const over an empty range
  {
    RangeView range(buff, buff);

    const std::ranges::enumerate_view view(range);

    auto it = view.end();
    assert(base(it.base()) == buff);
    assert(it == view.end());

    auto constIt = std::as_const(view).end();
    assert(base(constIt.base()) == buff);
    assert(constIt == std::as_const(view).end());
  }

  // end() over an 1-element range
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);

    auto it = view.end();
    assert(base(it.base()) == buff + 1);

    auto constIt = std::as_const(view).end();
    assert(base(constIt.base()) == buff + 1);
  }

  // end() over an N-element range
  {
    RangeView range(buff, buff + 8);

    std::ranges::enumerate_view view(range);

    auto it = view.end();
    assert(base(it.base()) == buff + 8);

    auto constIt = std::as_const(view).end();
    assert(base(constIt.base()) == buff + 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

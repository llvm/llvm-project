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

// constexpr auto size() requires sized_range<V>;
// constexpr auto size() const requires sized_range<const V>;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "test_concepts.h"
#include "types.h"

struct NonSizedRangeView : std::ranges::view_base {
  using iterator = forward_iterator<int*>;
  iterator begin() const;
  iterator end() const;
};

static_assert(!std::ranges::sized_range<NonSizedRangeView>);
static_assert(!std::ranges::sized_range<const NonSizedRangeView>);

static_assert(!HasMemberSize<std::ranges::enumerate_view<NonSizedRangeView>>);
static_assert(!HasMemberSize<const std::ranges::enumerate_view<NonSizedRangeView>>);

constexpr bool test() {
  int buffer[] = {1, 2, 3};

  // Non-const and const are sized
  {
    auto view = std::views::enumerate(buffer);
    assert(view.size() == 3);
    assert(std::as_const(view).size() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

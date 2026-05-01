//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// friend constexpr bool operator==(const iterator& x, default_sentinel_t);
//
// The sentinel for cartesian_product_view is `default_sentinel_t`. Equality is decided
// via __at_end(), which returns true if ANY underlying iterator equals its range's end.
// (For an empty inner range, that condition is met from begin() onward.)

#include <array>
#include <cassert>
#include <ranges>

#include "../../range_adaptor_types.h"

constexpr bool test() {
  std::array a{1, 2, 3};
  std::array b{10, 20};

  { // begin/end of a non-empty common range
    std::ranges::cartesian_product_view v(a);
    auto it = v.begin();
    auto en = std::default_sentinel;

    assert(it == it);
    assert(it != en);
    ++it;
    assert(it != en);
    ++it;
    assert(it != en);
    ++it; // moved past last element
    assert(it == en);
  }

  { // 2-range -- sentinel reached after iterating over the entire product
    std::ranges::cartesian_product_view v(a, b);
    auto it = v.begin();
    for (int i = 0; i < 6; ++i) {
      assert(it != std::default_sentinel);
      ++it;
    }
    assert(it == std::default_sentinel);
  }

  { // 2-range -- begin() compares equal to default_sentinel when an inner or outer range is empty
    std::ranges::empty_view<int> empty;
    std::ranges::cartesian_product_view v1(a, empty);
    assert(v1.begin() == std::default_sentinel);

    std::ranges::cartesian_product_view v2(empty, a);
    assert(v2.begin() == std::default_sentinel);

    std::ranges::cartesian_product_view v3(empty, empty);
    assert(v3.begin() == std::default_sentinel);
  }

  { // Non-common underlying first range -- view's end() returns default_sentinel, not an iterator
    // (cartesian-product-is-common is determined by the first range only.)
    std::ranges::cartesian_product_view v(InputNonCommonView{a}, ForwardSizedView{b});
    using View = decltype(v);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::same_as<decltype(v.end()), std::default_sentinel_t>);
    auto it = v.begin();
    for (int i = 0; i < 6; ++i) {
      assert(it != std::default_sentinel);
      ++it;
    }
    assert(it == std::default_sentinel);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

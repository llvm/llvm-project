//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

// Functional tests of std::ranges::cache_latest_view.

#include <cassert>
#include <ranges>
#include <vector>
#include <algorithm>

constexpr bool test() {
  // Motivational example from P3138R3.
  std::vector<int> vec          = {1, 2, 3, 4, 5};
  std::vector<int> filtered_vec = {4, 16};

  { // Uncached transform and filter.
    int transform_counter = 0;
    const auto get_square = [&](int i) {
      ++transform_counter;
      return i * i;
    };
    auto view = vec | std::views::transform(get_square) | std::views::filter([&](int i) { return i % 2 == 0; });

    assert(std::ranges::equal(view, filtered_vec));
    // The transform is called twice for each element that is not filtered out.
    assert(transform_counter == 7);
  }
  { // Cached transform and filter.
    int transform_counter = 0;
    const auto get_square = [&](int i) {
      ++transform_counter;
      return i * i;
    };
    auto view = vec | std::views::transform(get_square) | std::views::cache_latest |
                std::views::filter([&](int i) { return i % 2 == 0; });

    assert(std::ranges::equal(view, filtered_vec));
    // The transform is called only once for each element.
    assert(transform_counter == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

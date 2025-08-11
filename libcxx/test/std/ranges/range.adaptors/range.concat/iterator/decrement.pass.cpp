//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <cassert>
#include "test_macros.h"
#include "../types.h"

constexpr void test() {
  // Test with a single view
  {
    constexpr static std::array<int, 5> array{0, 1, 2, 3, 4};
    constexpr static std::ranges::concat_view view(std::views::all(array));
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end());

    auto& result = --it;
    ASSERT_SAME_TYPE(decltype(result)&, decltype(--it));
    assert(&result == &it);
    assert(result == view.begin() + 4);
  }

  // Test with more than one view
  {
    constexpr static std::array<int, 3> array{0, 1, 2};
    constexpr static std::array<int, 3> array1{3, 4, 5};
    constexpr std::ranges::concat_view view(std::views::all(array), std::views::all(array1));
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end());

    auto& result = --it;
    assert(&result == &it);

    --it;
    assert(*it == 4);
    assert(it == view.begin() + 4);
  }

  // Test going forward and then backward on the same iterator
  {
    constexpr static std::array<int, 5> array{0, 1, 2, 3, 4};
    constexpr static std::ranges::concat_view view(std::views::all(array));
    auto it = view.begin();
    ++it;
    --it;
    assert(*it == array[0]);
    ++it;
    ++it;
    --it;
    assert(*it == array[1]);
    ++it;
    ++it;
    --it;
    assert(*it == array[2]);
    ++it;
    ++it;
    --it;
    assert(*it == array[3]);
  }

  // Test post-decrement
  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    std::ranges::concat_view view(std::views::all(array));
    auto it = std::ranges::next(view.begin(), view.end());
    assert(it == view.end()); // test the test
    auto result = it--;
    ASSERT_SAME_TYPE(decltype(result), decltype(it--));
    assert(result == view.end());
    assert(it == (result - 1));
  }
}

int main(int, char**) {
  test();
  return 0;
}

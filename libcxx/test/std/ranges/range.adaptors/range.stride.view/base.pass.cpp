//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "test.h"
#include <cassert>
#include <ranges>

template <typename T>
concept can_call_base_on = requires(T t) { std::forward<T>(t).base(); };

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the const& overload
  {
    bool moved(false), copied(false);
    MovedCopiedTrackedBasicView range(buff, buff + 8, &moved, &copied);
    std::ranges::stride_view<MovedCopiedTrackedBasicView<int>> const view(std::move(range), 3);
    assert(moved);
    assert(!copied);
    std::same_as<MovedCopiedTrackedBasicView<int>> decltype(auto) result = view.base();
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Check the && overload
  {
    bool moved(false), copied(false);
    MovedCopiedTrackedBasicView range(buff, buff + 8, &moved, &copied);
    std::ranges::stride_view<MovedCopiedTrackedBasicView<int>> view(std::move(range), 3);
    assert(moved);
    assert(!copied);
    std::same_as<MovedCopiedTrackedBasicView<int>> decltype(auto) result = std::move(view).base();
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Check the && overload (again)
  {
    bool moved(false), copied(false);
    MovedCopiedTrackedBasicView range(buff, buff + 8, &moved, &copied);
    std::same_as<MovedCopiedTrackedBasicView<int>> decltype(auto) result =
        std::ranges::stride_view<MovedCopiedTrackedBasicView<int>>(std::move(range), 3).base();
    assert(moved);
    assert(!copied);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Ensure the const& overload is not considered when the base is not copy-constructible
  {
    static_assert(!can_call_base_on<std::ranges::stride_view<MovedOnlyTrackedBasicView<>> const&>);
    static_assert(!can_call_base_on<std::ranges::stride_view<MovedOnlyTrackedBasicView<>>&>);
    static_assert(can_call_base_on<std::ranges::stride_view<MovedOnlyTrackedBasicView<>>&&>);
    static_assert(can_call_base_on<std::ranges::stride_view<MovedOnlyTrackedBasicView<>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

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
    Range range(buff, buff + 8);
    std::ranges::stride_view<Range<int>> const view(range, 3);
    std::same_as<Range<int>> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Check the && overload
  {
    Range range(buff, buff + 8);
    std::ranges::stride_view<Range<int>> view(range, 3);
    std::same_as<Range<int>> decltype(auto) result = std::move(view).base();
    assert(result.wasMoveInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Check the && overload (again)
  {
    Range range(buff, buff + 8);
    std::same_as<Range<int>> decltype(auto) result = std::ranges::stride_view<Range<int>>(range, 3).base();
    assert(result.wasMoveInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Ensure the const& overload is not considered when the base is not copy-constructible
  {
    static_assert(!can_call_base_on<std::ranges::stride_view<NoCopyRange> const&>);
    static_assert(!can_call_base_on<std::ranges::stride_view<NoCopyRange>&>);
    static_assert(can_call_base_on<std::ranges::stride_view<NoCopyRange>&&>);
    static_assert(can_call_base_on<std::ranges::stride_view<NoCopyRange>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

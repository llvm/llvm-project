//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size() const requires (!same_as<Bound, unreachable_sentinel_t>);

#include <cassert>
#include <iterator>
#include <limits>
#include <ranges>

template <class T>
concept has_size = requires(T&& view) {
  { std::forward<T>(view).size() };
};

static_assert(has_size<std::ranges::repeat_view<int, int>>);
static_assert(!has_size<std::ranges::repeat_view<int>>);
static_assert(!has_size<std::ranges::repeat_view<int, std::unreachable_sentinel_t>>);

constexpr bool test() {
  {
    std::ranges::repeat_view<int, int> rv(10, 20);
    assert(rv.size() == 20);
  }

  {
    constexpr int int_max = std::numeric_limits<int>::max();
    std::ranges::repeat_view<int, int> rv(10, int_max);
    assert(rv.size() == int_max);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

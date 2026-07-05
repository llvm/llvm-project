//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end();

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

// Test that end is not const
template <class T>
concept HasEnd = requires(T t) { t.end(); };

static_assert(HasEnd<std::ranges::split_view<View, View>>);
static_assert(!HasEnd<const std::ranges::split_view<View, View>>);

constexpr bool test() {
  // return iterator
  {
    int buffer[]   = {1, 2, -1, 4, 5, 6, 5, 4, -1, 2, 1};
    auto inputView = std::views::all(buffer);
    static_assert(std::ranges::common_range<decltype(inputView)>);

    std::ranges::split_view sv(buffer, -1);
    using SplitIter                           = std::ranges::iterator_t<decltype(sv)>;
    std::same_as<SplitIter> decltype(auto) sentinel = sv.end();
    assert(sentinel.base() == buffer + 11);
  }

  // return sentinel
  {
    using Iter   = int*;
    using Sent   = sentinel_wrapper<Iter>;
    using Range  = std::ranges::subrange<Iter, Sent>;
    int buffer[] = {1, 2, -1, 4, 5, 6, 5, 4, -1, 2, 1};
    Range range  = {buffer, Sent{buffer + 11}};
    static_assert(!std::ranges::common_range<Range>);

    std::ranges::split_view sv(range, -1);
    auto sentinel = sv.end();

    using SplitIter = std::ranges::iterator_t<decltype(sv)>;
    static_assert(!std::same_as<decltype(sentinel), SplitIter>);

    assert(std::next(sv.begin(), 3) == sentinel);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

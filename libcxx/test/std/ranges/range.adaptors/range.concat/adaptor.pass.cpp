//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::filter

#include <ranges>

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_range.h"

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

template <typename View>
constexpr void compareViews(View v, std::initializer_list<int> list) {
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2; ++b1, ++b2) {
    assert(*b1 == *b2);
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

constexpr bool test() {
  int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    using Result = std::ranges::concat_view<Range>;
    Range range(buff, buff + 8);

    {
      decltype(auto) result = std::views::concat(range);
      compareViews(result, {0, 1, 2, 3, 4, 5, 6, 7});
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

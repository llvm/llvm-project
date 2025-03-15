//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <cassert>
#include <concepts>
#include <type_traits>
#include <iterator>
#include "test_iterators.h"

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

struct CommonRange : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  constexpr explicit CommonRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Iterator end() const { return Iterator(end_); }

private:
  int* begin_;
  int* end_;
};

struct NotCommonRange : std::ranges::view_base {
  constexpr explicit NotCommonRange() {}
  char* begin();
  bool end();
};

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.end()`
  {
    Range range(buff, buff + 1);
    std::ranges::concat_view view(range);
    using ConcatSentinel = std::ranges::sentinel_t<decltype(view)>;
    ASSERT_SAME_TYPE(ConcatSentinel, decltype(view.end()));
  }

  // Check a not a common range
  {
    Range range(buff, buff + 1);
    std::ranges::concat_view view(range);
    using ConcatSentinel = std::default_sentinel_t;
    ASSERT_SAME_TYPE(ConcatSentinel, decltype(view.end()));
  }

  // end() on an empty range
  {
    Range range(buff, buff);
    std::ranges::concat_view view(range);
    auto begin = view.begin();
    auto end   = view.end();
    assert(begin == end);
  }

  // end() on a common_range
  {
    CommonRange range(buff, buff + 1);
    CommonRange range_2(buff + 2, buff + 3);
    std::ranges::concat_view view(range, range_2);
    auto it = view.begin();
    it++;
    it++;
    auto end = view.end();
    assert(it == end);
    static_assert(std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  return true;
}

int main(int, char**) {
  test();

  return 0;
}

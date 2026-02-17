//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr auto end();

#include <ranges>

#include <cassert>
#include <concepts>
#include <functional>

#include "test_iterators.h"

struct NonCommonRange : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit NonCommonRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

static_assert(std::ranges::forward_range<NonCommonRange>);
static_assert(!std::ranges::common_range<NonCommonRange>);

struct CommonRange : std::ranges::view_base {
  using Iterator = bidirectional_iterator<int*>;
  constexpr explicit CommonRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Iterator end() const { return Iterator(end_); }

private:
  int* begin_;
  int* end_;
};

static_assert(std::ranges::bidirectional_range<CommonRange>);
static_assert(std::ranges::common_range<CommonRange>);

constexpr bool test() {
  int buff[] = {1, 0, 3, 1, 2, 3, 4, 5};

  // Check the return type of `end()`
  {
    CommonRange range(buff, buff + 1);
    auto pred = [](int, int) { return true; };
    std::ranges::chunk_by_view view(range, pred);
    using ChunkByView = decltype(view);
    static_assert(std::ranges::common_range<ChunkByView>);
    ASSERT_SAME_TYPE(std::ranges::sentinel_t<ChunkByView>, decltype(view.end()));
  }

  // end() on an empty range
  {
    CommonRange range(buff, buff);
    auto pred = [](int x, int y) { return x <= y; };
    std::ranges::chunk_by_view view(range, pred);
    auto end = view.end();
    assert(end == std::default_sentinel);
  }

  // end() on a 1-element range
  {
    CommonRange range(buff, buff + 1);
    auto pred = [](int& x, int& y) { return x <= y; };
    std::ranges::chunk_by_view view(range, pred);
    auto end = view.end();
    assert(base((*--end).begin()) == buff);
    assert(base((*end).end()) == buff + 1);
  }

  // end() on a 2-element range
  {
    CommonRange range(buff, buff + 2);
    auto pred = [](int const& x, int const& y) { return x <= y; };
    std::ranges::chunk_by_view view(range, pred);
    auto end = view.end();
    assert(base((*--end).begin()) == buff + 1);
    assert(base((*--end).end()) == buff + 1);
  }

  // end() on a 8-element range
  {
    CommonRange range(buff, buff + 8);
    auto pred = [](const int x, const int y) { return x < y; };
    std::ranges::chunk_by_view view(range, pred);
    auto end = view.end();
    assert(base((*--end).end()) == buff + 8);
    assert(base((*--end).end()) == buff + 3);
  }

  // end() on a non-common range
  {
    NonCommonRange range(buff, buff + 1);
    std::ranges::chunk_by_view view(range, std::ranges::less_equal{});
    auto end = view.end();
    ASSERT_SAME_TYPE(std::default_sentinel_t, std::ranges::sentinel_t<decltype(view)>);
    ASSERT_SAME_TYPE(std::default_sentinel_t, decltype(end));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

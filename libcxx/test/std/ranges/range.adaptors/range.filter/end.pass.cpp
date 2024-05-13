//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end();

#include <ranges>

#include <cassert>
#include <concepts>
#include <type_traits>
#include "test_iterators.h"

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct CommonRange : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  constexpr explicit CommonRange(int* b, int* e) : begin_(b), end_(e) { }
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Iterator end() const { return Iterator(end_); }

private:
  int* begin_;
  int* end_;
};

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.end()`
  {
    Range range(buff, buff + 1);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    using FilterSentinel = std::ranges::sentinel_t<decltype(view)>;
    ASSERT_SAME_TYPE(FilterSentinel, decltype(view.end()));
  }

  // end() on an empty range
  {
    Range range(buff, buff);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff);
  }

  // end() on a 1-element range
  {
    Range range(buff, buff + 1);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff + 1);
    static_assert(!std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  // end() on a 2-element range
  {
    Range range(buff, buff + 2);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff + 2);
    static_assert(!std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  // end() on a N-element range
  {
    for (int k = 1; k != 8; ++k) {
      Range range(buff, buff + 8);
      auto pred = [=](int i) { return i == k; };
      std::ranges::filter_view view(range, pred);
      auto end = view.end();
      assert(base(base(end.base())) == buff + 8);
      static_assert(!std::is_same_v<decltype(end), decltype(view.begin())>);
    }
  }

  // end() on a common_range
  {
    CommonRange range(buff, buff + 8);
    auto pred = [](int i) { return i % 2 == 0; };
    std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(end.base()) == buff + 8);
    static_assert(std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

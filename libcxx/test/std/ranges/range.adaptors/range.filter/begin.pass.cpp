//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr iterator begin();

#include <ranges>

#include <cassert>
#include "test_iterators.h"
#include "types.h"

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

// A range that isn't a forward_range, used to test filter_view
// when we don't cache the result of begin()
struct InputRange : std::ranges::view_base {
  using Iterator = cpp17_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit InputRange(int* b, int* e) : begin_(b), end_(e) { }
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct TrackingPred : TrackInitialization {
  using TrackInitialization::TrackInitialization;
  constexpr bool operator()(int i) const { return i % 2 == 0; }
};

template <typename Range>
constexpr void general_tests() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.begin()`
  {
    Range range(buff, buff + 1);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    using FilterIterator = std::ranges::iterator_t<decltype(view)>;
    ASSERT_SAME_TYPE(FilterIterator, decltype(view.begin()));
  }

  // begin() over an empty range
  {
    Range range(buff, buff);
    auto pred = [](int) { return true; };
    std::ranges::filter_view view(range, pred);
    auto it = view.begin();
    assert(base(it.base()) == buff);
    assert(it == view.end());
  }

  // begin() over a 1-element range
  {
    {
      Range range(buff, buff + 1);
      auto pred = [](int i) { return i == 1; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff);
    }
    {
      Range range(buff, buff + 1);
      auto pred = [](int) { return false; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff + 1);
      assert(it == view.end());
    }
  }

  // begin() over a 2-element range
  {
    {
      Range range(buff, buff + 2);
      auto pred = [](int i) { return i == 1; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff);
    }
    {
      Range range(buff, buff + 2);
      auto pred = [](int i) { return i == 2; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff + 1);
    }
    {
      Range range(buff, buff + 2);
      auto pred = [](int) { return false; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff + 2);
      assert(it == view.end());
    }
  }

  // begin() over a N-element range
  {
    for (int k = 1; k != 8; ++k) {
      Range range(buff, buff + 8);
      auto pred = [=](int i) { return i == k; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff + (k - 1));
    }
    {
      Range range(buff, buff + 8);
      auto pred = [](int) { return false; };
      std::ranges::filter_view view(range, pred);
      auto it = view.begin();
      assert(base(it.base()) == buff + 8);
      assert(it == view.end());
    }
  }

  // Make sure we do not make a copy of the predicate when we call begin()
  // (we should be passing it to ranges::find_if using std::ref)
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 2);
    std::ranges::filter_view view(range, TrackingPred(&moved, &copied));
    moved = false;
    copied = false;
    [[maybe_unused]] auto it = view.begin();
    assert(!moved);
    assert(!copied);
  }

  // Test with a non-const predicate
  {
    Range range(buff, buff + 8);
    auto pred = [](int i) mutable { return i % 2 == 0; };
    std::ranges::filter_view view(range, pred);
    auto it = view.begin();
    assert(base(it.base()) == buff + 1);
  }

  // Test with a predicate that takes by non-const reference
  {
    Range range(buff, buff + 8);
    auto pred = [](int& i) { return i % 2 == 0; };
    std::ranges::filter_view view(range, pred);
    auto it = view.begin();
    assert(base(it.base()) == buff + 1);
  }
}

template <typename ForwardRange>
constexpr void cache_tests() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Make sure that we cache the result of begin() on subsequent calls
  // (only applies to forward_ranges)
  ForwardRange range(buff, buff + 8);
  int called = 0;
  auto pred = [&](int i) { ++called; return i == 3; };

  std::ranges::filter_view view(range, pred);
  assert(called == 0);
  for (int k = 0; k != 3; ++k) {
    auto it = view.begin();
    assert(base(it.base()) == buff + 2);
    assert(called == 3);
  }
}

constexpr bool test() {
  general_tests<Range>();
  general_tests<InputRange>(); // test when we don't cache the result
  cache_tests<Range>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

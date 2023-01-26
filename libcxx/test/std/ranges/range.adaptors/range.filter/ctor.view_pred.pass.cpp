//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr filter_view(View, Pred);

#include <ranges>

#include <cassert>
#include <utility>
#include "types.h"

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

struct Pred {
  constexpr bool operator()(int i) const { return i % 2 != 0; }
};

struct TrackingPred : TrackInitialization {
  using TrackInitialization::TrackInitialization;
  constexpr bool operator()(int) const;
};

struct TrackingRange : TrackInitialization, std::ranges::view_base {
  using TrackInitialization::TrackInitialization;
  int* begin() const;
  int* end() const;
};

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test explicit syntax
  {
    Range range(buff, buff + 8);
    Pred pred;
    std::ranges::filter_view<Range, Pred> view(range, pred);
    auto it = view.begin(), end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 3);
    assert(*it++ == 5);
    assert(*it++ == 7);
    assert(it == end);
  }

  // Test implicit syntax
  {
    Range range(buff, buff + 8);
    Pred pred;
    std::ranges::filter_view<Range, Pred> view = {range, pred};
    auto it = view.begin(), end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 3);
    assert(*it++ == 5);
    assert(*it++ == 7);
    assert(it == end);
  }

  // Make sure we move the view
  {
    bool moved = false, copied = false;
    TrackingRange range(&moved, &copied);
    Pred pred;
    [[maybe_unused]] std::ranges::filter_view<TrackingRange, Pred> view(std::move(range), pred);
    assert(moved);
    assert(!copied);
  }

  // Make sure we move the predicate
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 8);
    TrackingPred pred(&moved, &copied);
    [[maybe_unused]] std::ranges::filter_view<Range, TrackingPred> view(range, std::move(pred));
    assert(moved);
    assert(!copied);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

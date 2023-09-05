//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr chunk_by_view(View, Pred);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <utility>

#include "types.h"

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

static_assert(std::ranges::view<Range>);
static_assert(std::ranges::forward_range<Range>);

struct Pred {
  constexpr bool operator()(int x, int y) const { return x <= y; }
};

struct TrackingPred : TrackInitialization {
  using TrackInitialization::TrackInitialization;
  constexpr bool operator()(int&, int&) const;
};

struct TrackingRange : TrackInitialization, std::ranges::view_base {
  using TrackInitialization::TrackInitialization;
  int* begin() const;
  int* end() const;
};

template <class T>
void implicitConstructionTest(T);

template <class T, class... Args>
concept ImplicitConstructibleFrom =
    requires(Args&&... args) { implicitConstructionTest({std::forward<Args>(args)...}); };

constexpr bool test() {
  int buff[] = {1, 2, 3, 0, 1, 2, -1, -1, 0};

  // Test explicit syntax
  {
    Range range(buff, buff + 9);
    Pred pred;
    std::ranges::chunk_by_view<Range, Pred> view(range, pred);
    auto it = view.begin(), end = view.end();
    assert(std::ranges::equal(*it++, std::array{1, 2, 3}));
    assert(std::ranges::equal(*it++, std::array{0, 1, 2}));
    assert(std::ranges::equal(*it++, std::array{-1, -1, 0}));
    assert(it == end);
  }

  // Test implicit syntax
  {
    using ChunkByView = std::ranges::chunk_by_view<Range, Pred>;
    static_assert(!ImplicitConstructibleFrom<ChunkByView, Range, Pred>);
    static_assert(!ImplicitConstructibleFrom<ChunkByView, const Range&, const Pred&>);
  }

  // Make sure we move the view
  {
    bool moved = false, copied = false;
    TrackingRange range(&moved, &copied);
    Pred pred;
    [[maybe_unused]] std::ranges::chunk_by_view<TrackingRange, Pred> view(std::move(range), pred);
    assert(moved);
    assert(!copied);
  }

  // Make sure we move the predicate
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 9);
    TrackingPred pred(&moved, &copied);
    [[maybe_unused]] std::ranges::chunk_by_view<Range, TrackingPred> view(range, std::move(pred));
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

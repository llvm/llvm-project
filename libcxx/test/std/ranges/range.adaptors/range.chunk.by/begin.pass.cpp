//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr iterator begin();

#include <ranges>

#include <cassert>
#include <utility>

#include "test_iterators.h"
#include "types.h"

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

struct TrackingPred : TrackInitialization {
  using TrackInitialization::TrackInitialization;
  constexpr bool operator()(int x, int y) { return x != -y; }
};

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

static_assert(HasBegin<std::ranges::chunk_by_view<Range, TrackingPred>>);
static_assert(!HasBegin<const std::ranges::chunk_by_view<Range, TrackingPred>>);

constexpr bool test() {
  int buff[] = {-4, -3, -2, -1, 1, 2, 3, 4};

  // Check the return type of `begin()`
  {
    Range range(buff, buff + 1);
    auto pred = [](int, int) { return true; };
    std::ranges::chunk_by_view view(range, pred);
    using ChunkByIterator = std::ranges::iterator_t<decltype(view)>;
    ASSERT_SAME_TYPE(ChunkByIterator, decltype(view.begin()));
  }

  // begin() over an empty range
  {
    Range range(buff, buff);
    auto pred = [](int, int) { return true; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(it == view.begin());
    assert(it == view.end());
  }

  // begin() over a 1-element range
  {
    Range range(buff, buff + 1);
    auto pred = [](int x, int y) { return x == y; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(base((*it).begin()) == buff);
    assert(base((*it).end()) == buff + 1);
  }

  // begin() over a 2-element range
  {
    Range range(buff, buff + 2);
    auto pred = [](int x, int y) { return x == y; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(base((*it).begin()) == buff);
    assert(base((*it).end()) == buff + 1);
    assert(base((*++it).begin()) == buff + 1);
    assert(base((*it).end()) == buff + 2);
  }

  // begin() over a longer range
  {
    Range range(buff, buff + 8);
    auto pred = [](int x, int y) { return x != -y; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(base((*it).end()) == buff + 4);
  }

  // Make sure we do not make a copy of the predicate when we call begin()
  // (we should be passing it to ranges::adjacent_find using std::ref)
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 2);
    std::ranges::chunk_by_view view(range, TrackingPred(&moved, &copied));
    std::exchange(moved, false);
    [[maybe_unused]] auto it = view.begin();
    assert(!moved);
    assert(!copied);
  }

  // Test with a non-const predicate
  {
    Range range(buff, buff + 8);
    auto pred = [](int x, int y) mutable { return x != -y; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(base((*it).end()) == buff + 4);
  }

  // Test with a predicate that takes by non-const reference
  {
    Range range(buff, buff + 8);
    auto pred = [](int& x, int& y) { return x != -y; };
    std::ranges::chunk_by_view view(range, pred);
    auto it = view.begin();
    assert(base((*it).end()) == buff + 4);
  }

  // Test caching
  {
    // Make sure that we cache the result of begin() on subsequent calls
    Range range(buff, buff + 8);
    int called = 0;
    auto pred  = [&](int x, int y) {
      ++called;
      return x != -y;
    };

    std::ranges::chunk_by_view view(range, pred);
    assert(called == 0);
    for (int k = 0; k != 3; ++k) {
      auto it = view.begin();
      assert(base((*it).end()) == buff + 4);
      assert(called == 4); // 4, because the cached iterator is 'buff + 4' (end of the first chunk)
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

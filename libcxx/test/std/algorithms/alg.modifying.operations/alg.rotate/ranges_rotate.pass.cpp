//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<permutable I, sentinel_for<I> S>
//   constexpr subrange<I> rotate(I first, I middle, S last);                                        // since C++20
//
// template<forward_range R>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R> rotate(R&& r, iterator_t<R> middle);                           // Since C++20

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "type_algorithms.h"

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*>
concept HasRotateIter = requires(Iter&& iter, Sent&& sent) {
  std::ranges::rotate(std::forward<Iter>(iter), std::forward<Iter>(iter), std::forward<Sent>(sent));
};

static_assert(HasRotateIter<int*, int*>);

// !permutable<I>
static_assert(!HasRotateIter<PermutableNotForwardIterator>);
static_assert(!HasRotateIter<PermutableNotSwappable>);

// !sentinel_for<S, I>
static_assert(!HasRotateIter<int*, SentinelForNotSemiregular>);
static_assert(!HasRotateIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// Test constraints of the (range) overload.
// =========================================

template <class Range>
concept HasRotateRange = requires(Range&& range, std::ranges::iterator_t<Range> iter) {
  std::ranges::rotate(std::forward<Range>(range), iter);
};

template <class T>
using R = UncheckedRange<T>;

static_assert(HasRotateRange<R<int*>>);

// !forward_range<R>
static_assert(!HasRotateRange<ForwardRangeNotDerivedFrom>);
static_assert(!HasRotateRange<ForwardRangeNotIncrementable>);
static_assert(!HasRotateRange<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasRotateRange<ForwardRangeNotSentinelEqualityComparableWith>);

// !permutable<iterator_t<R>>
static_assert(!HasRotateRange<PermutableRangeNotForwardIterator>);
static_assert(!HasRotateRange<PermutableRangeNotSwappable>);

template <class Iter, class Sent, std::size_t N>
constexpr void test_one(const std::array<int, N> input, std::size_t mid_index, std::array<int, N> expected) {
  assert(mid_index <= N);

  { // (iterator, sentinel) overload.
    auto in    = input;
    auto begin = Iter(in.data());
    auto mid   = Iter(in.data() + mid_index);
    auto end   = Sent(Iter(in.data() + in.size()));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::rotate(begin, mid, end);
    assert(base(result.begin()) == in.data() + in.size() - mid_index);
    assert(base(result.end()) == in.data() + in.size());
    assert(in == expected);
  }

  { // (range) overload.
    auto in    = input;
    auto begin = Iter(in.data());
    auto mid   = Iter(in.data() + mid_index);
    auto end   = Sent(Iter(in.data() + in.size()));
    auto range = std::ranges::subrange(std::move(begin), std::move(end));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::rotate(range, mid);
    assert(base(result.begin()) == in.data() + in.size() - mid_index);
    assert(base(result.end()) == in.data() + in.size());
    assert(in == expected);
  }
}

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  // Empty sequence.
  test_one<Iter, Sent, 0>({}, 0, {});

  // 1-element sequence.
  test_one<Iter, Sent, 1>({1}, 0, {1});

  // 2-element sequence.
  test_one<Iter, Sent, 2>({1, 2}, 1, {2, 1});

  // 3-element sequence.
  test_one<Iter, Sent, 3>({1, 2, 3}, 1, {2, 3, 1});
  test_one<Iter, Sent, 3>({1, 2, 3}, 2, {3, 1, 2});

  // Longer sequence.
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 2, {3, 4, 5, 6, 7, 1, 2});

  // Rotate around the middle.
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 3, {4, 5, 6, 7, 1, 2, 3});

  // Rotate around the 1st element (no-op).
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 0, {1, 2, 3, 4, 5, 6, 7});

  // Rotate around the 2nd element.
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 1, {2, 3, 4, 5, 6, 7, 1});

  // Rotate around the last element.
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 6, {7, 1, 2, 3, 4, 5, 6});

  // Pass `end()` as `mid` (no-op).
  test_one<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 7, {1, 2, 3, 4, 5, 6, 7});
}

constexpr bool test() {
  types::for_each(types::forward_iterator_list<int*>(), []<class Iter>() {
    test_iter_sent<Iter, Iter>();
    test_iter_sent<Iter, sentinel_wrapper<Iter>>();
  });

  { // Complexity: at most `last - first` swaps.
    const std::array input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto expected          = static_cast<int>(input.size());

    {
      auto in    = input;
      int swaps  = 0;
      auto begin = adl::Iterator::TrackSwaps(in.data(), swaps);
      auto end   = adl::Iterator::TrackSwaps(in.data() + in.size(), swaps);

      for (std::size_t mid = 0; mid != input.size(); ++mid) {
        std::ranges::rotate(begin, begin + mid, end);
        assert(swaps <= expected);
      }
    }

    {
      auto in    = input;
      int swaps  = 0;
      auto begin = adl::Iterator::TrackSwaps(in.data(), swaps);
      auto end   = adl::Iterator::TrackSwaps(in.data() + in.size(), swaps);
      auto range = std::ranges::subrange(begin, end);

      for (std::size_t mid = 0; mid != input.size(); ++mid) {
        std::ranges::rotate(range, begin + mid);
        assert(swaps <= expected);
      }
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

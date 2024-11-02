//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_strict_weak_order<projected<I, Proj>> Comp = ranges::less>
//   constexpr bool is_heap(I first, S last, Comp comp = {}, Proj proj = {});                   // Since C++20
//
// template<random_access_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   constexpr bool is_heap(R&& r, Comp comp = {}, Proj proj = {});                             // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Comp = std::ranges::less>
concept HasIsHeapIter =
    requires(Iter&& iter, Sent&& sent, Comp&& comp) {
      std::ranges::is_heap(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Comp>(comp));
    };

static_assert(HasIsHeapIter<int*, int*, std::ranges::less>);

// !random_access_iterator<I>
static_assert(!HasIsHeapIter<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasIsHeapIter<RandomAccessIteratorBadIndex>);

// !sentinel_for<S, I>
static_assert(!HasIsHeapIter<int*, SentinelForNotSemiregular>);
static_assert(!HasIsHeapIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

struct NoComparator {};
// !indirect_strict_weak_order<Comp, projected<I, Proj>>
static_assert(!HasIsHeapIter<NoComparator*, NoComparator*>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Comp = std::ranges::less>
concept HasIsHeapRange =
    requires(Range&& range, Comp&& comp) {
      std::ranges::is_heap(std::forward<Range>(range), std::forward<Comp>(comp));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasIsHeapRange<R<int*>>);

// !random_access_range<R>
static_assert(!HasIsHeapRange<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasIsHeapRange<RandomAccessRangeBadIndex>);

// !indirect_strict_weak_order<Comp, projected<iterator_t<R>, Proj>>
static_assert(!HasIsHeapRange<R<NoComparator*>>);

template <class Iter, class Sent, size_t N>
constexpr void test_one(std::array<int, N> input, bool expected) {
  auto begin = Iter(input.data());
  auto end = Sent(Iter(input.data() + input.size()));

  { // (iterator, sentinel) overload.
    std::same_as<bool> decltype(auto) result = std::ranges::is_heap(begin, end);
    assert(result == expected);
  }

  { // (range) overload.
    auto range = std::ranges::subrange(begin, end);
    std::same_as<bool> decltype(auto) result = std::ranges::is_heap(range);
    assert(result == expected);
  }
}

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  // Empty sequence.
  test_one<Iter, Sent, 0>({}, true);
  // 1-element sequence.
  test_one<Iter, Sent>(std::array{1}, true);
  // 2-element sequence, a heap.
  test_one<Iter, Sent>(std::array{2, 1}, true);
  // 2-element sequence, not a heap.
  test_one<Iter, Sent>(std::array{1, 2}, false);
  // Longer sequence, a heap.
  test_one<Iter, Sent>(std::array{8, 6, 7, 3, 4, 1, 5, 2}, true);
  // Longer sequence, not a heap.
  test_one<Iter, Sent>(std::array{8, 6, 7, 3, 4, 1, 2, 5}, false);
  // Longer sequence with duplicates, a heap.
  test_one<Iter, Sent>(std::array{8, 7, 5, 5, 6, 4, 1, 2, 3, 2}, true);
  // Longer sequence with duplicates, not a heap.
  test_one<Iter, Sent>(std::array{7, 5, 5, 6, 4, 1, 2, 3, 2, 8}, false);
  // All elements are the same.
  test_one<Iter, Sent>(std::array{1, 1, 1}, true);
}

template <class Iter>
constexpr void test_iter() {
  test_iter_sent<Iter, Iter>();
  test_iter_sent<Iter, sentinel_wrapper<Iter>>();
}

constexpr void test_iterators() {
  test_iter<random_access_iterator<int*>>();
  test_iter<contiguous_iterator<int*>>();
  test_iter<int*>();
  test_iter<const int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom comparator works.
    std::ranges::less ls;
    std::ranges::greater gt;
    std::array in = {1, 3, 2, 5, 4, 7, 8, 6};

    { // (iterator, sentinel) overload.
      assert(!std::ranges::is_heap(in.begin(), in.end(), ls));
      assert(std::ranges::is_heap(in.begin(), in.end(), gt));
    }

    { // (range) overload.
      assert(!std::ranges::is_heap(in, ls));
      assert(std::ranges::is_heap(in, gt));
    }
  }

  { // A custom projection works.
    struct A {
      int x;
      constexpr auto operator<=>(const A&) const = default;
    };

    std::array in = {A{-8}, A{-6}, A{-7}, A{-3}, A{-4}, A{-1}, A{-5}, A{-2}};
    auto negate = [](A a) { return a.x * -1; };

    { // (iterator, sentinel) overload.
      assert(!std::ranges::is_heap(in.begin(), in.end(), {}));
      assert(std::ranges::is_heap(in.begin(), in.end(), {}, negate));
    }

    { // (range) overload.
      assert(!std::ranges::is_heap(in, {}));
      assert(std::ranges::is_heap(in, {}, negate));
    }
  }


  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

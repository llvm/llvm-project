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
//   constexpr I is_heap_until(I first, S last, Comp comp = {}, Proj proj = {});                // Since C++20
//
// template<random_access_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   constexpr borrowed_iterator_t<R>
//     is_heap_until(R&& r, Comp comp = {}, Proj proj = {});                                    // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Comp = std::ranges::less>
concept HasIsHeapUntilIter =
    requires(Iter&& iter, Sent&& sent, Comp&& comp) {
      std::ranges::is_heap_until(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Comp>(comp));
    };

static_assert(HasIsHeapUntilIter<int*, int*, std::ranges::less>);

// !random_access_iterator<I>
static_assert(!HasIsHeapUntilIter<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasIsHeapUntilIter<RandomAccessIteratorBadIndex>);

// !sentinel_for<S, I>
static_assert(!HasIsHeapUntilIter<int*, SentinelForNotSemiregular>);
static_assert(!HasIsHeapUntilIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

struct NoComparator {};
// !indirect_strict_weak_order<Comp, projected<I, Proj>>
static_assert(!HasIsHeapUntilIter<NoComparator*, NoComparator*>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Comp = std::ranges::less>
concept HasIsHeapUntilRange =
    requires(Range&& range, Comp&& comp) {
      std::ranges::is_heap_until(std::forward<Range>(range), std::forward<Comp>(comp));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasIsHeapUntilRange<R<int*>>);

// !random_access_range<R>
static_assert(!HasIsHeapUntilRange<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasIsHeapUntilRange<RandomAccessRangeBadIndex>);

// !indirect_strict_weak_order<Comp, projected<iterator_t<R>, Proj>>
static_assert(!HasIsHeapUntilRange<R<NoComparator*>>);

template <class Iter, class Sent, size_t N>
constexpr void test_one(std::array<int, N> input, size_t until_index) {
  auto begin = Iter(input.data());
  auto end = Sent(Iter(input.data() + input.size()));

  { // (iterator, sentinel) overload.
    std::same_as<Iter> decltype(auto) result = std::ranges::is_heap_until(begin, end);
    assert(base(result) == input.data() + until_index);
  }

  { // (range) overload.
    auto range = std::ranges::subrange(begin, end);
    std::same_as<Iter> decltype(auto) result = std::ranges::is_heap_until(range);
    assert(base(result) == input.data() + until_index);
  }
}

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  // Empty sequence.
  test_one<Iter, Sent, 0>({}, 0);
  // 1-element sequence.
  test_one<Iter, Sent>(std::array{1}, 1);
  // 2-element sequence, a heap.
  test_one<Iter, Sent>(std::array{2, 1}, 2);
  // 2-element sequence, not a heap.
  test_one<Iter, Sent>(std::array{1, 2}, 1);
  // Longer sequence, a heap.
  test_one<Iter, Sent>(std::array{8, 6, 7, 3, 4, 1, 5, 2}, 8);
  // Longer sequence, not a heap.
  test_one<Iter, Sent>(std::array{8, 6, 7, 3, 4, 1, 2, 5}, 7);
  // Longer sequence with duplicates, a heap.
  test_one<Iter, Sent>(std::array{8, 7, 5, 5, 6, 4, 1, 2, 3, 2}, 10);
  // Longer sequence with duplicates, not a heap.
  test_one<Iter, Sent>(std::array{7, 5, 5, 6, 4, 1, 2, 3, 2, 8}, 3);
  // All elements are the same.
  test_one<Iter, Sent>(std::array{1, 1, 1}, 3);
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
      auto result_default_comp = std::ranges::is_heap_until(in.begin(), in.end(), ls);
      assert(result_default_comp == in.begin() + 1);
      auto result_custom_comp = std::ranges::is_heap_until(in.begin(), in.end(), gt);
      assert(result_custom_comp == in.end());
    }

    { // (range) overload.
      auto result_default_comp = std::ranges::is_heap_until(in, ls);
      assert(result_default_comp == in.begin() + 1);
      auto result_custom_comp = std::ranges::is_heap_until(in, gt);
      assert(result_custom_comp == in.end());
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
      auto result_default_comp = std::ranges::is_heap_until(in.begin(), in.end(), {});
      assert(result_default_comp == in.begin() + 1);
      auto result_custom_comp = std::ranges::is_heap_until(in.begin(), in.end(), {}, negate);
      assert(result_custom_comp == in.end());
    }

    { // (range) overload.
      auto result_default_comp = std::ranges::is_heap_until(in, {});
      assert(result_default_comp == in.begin() + 1);
      auto result_custom_comp = std::ranges::is_heap_until(in, {}, negate);
      assert(result_custom_comp == in.end());
    }
  }


  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

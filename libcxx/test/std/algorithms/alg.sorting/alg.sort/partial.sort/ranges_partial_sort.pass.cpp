//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//           class Proj = identity>
//    requires sortable<I, Comp, Proj>
//    constexpr I
//      ranges::partial_sort(I first, I middle, S last, Comp comp = {}, Proj proj = {});      // since C++20
//
//  template<random_access_range R, class Comp = ranges::less, class Proj = identity>
//    requires sortable<iterator_t<R>, Comp, Proj>
//    constexpr borrowed_iterator_t<R>
//      ranges::partial_sort(R&& r, iterator_t<R> middle, Comp comp = {}, Proj proj = {});    // since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

// SFINAE tests.

using BadComparator = ComparatorNotCopyable<int*>;
static_assert(!std::sortable<int*, BadComparator>);

template <class Iter, class Sent = sentinel_wrapper<Iter>, class Comp = std::ranges::less>
concept HasPartialSortIt = requires(Iter first, Iter mid, Sent last, Comp comp) {
  std::ranges::partial_sort(first, mid, last, comp);
};

static_assert(HasPartialSortIt<int*>);
static_assert(!HasPartialSortIt<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasPartialSortIt<RandomAccessIteratorBadIndex>);
static_assert(!HasPartialSortIt<int*, SentinelForNotSemiregular>);
static_assert(!HasPartialSortIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasPartialSortIt<int*, int*, BadComparator>);
static_assert(!HasPartialSortIt<const int*>); // Doesn't satisfy `sortable`.

template <class Range, class Comp = std::ranges::less>
concept HasPartialSortR = requires(Range range, std::ranges::iterator_t<Range> mid, Comp comp) {
  std::ranges::partial_sort(range, mid, comp);
};

static_assert(HasPartialSortR<UncheckedRange<int*>>);
static_assert(!HasPartialSortR<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasPartialSortR<RandomAccessRangeBadIndex>);
static_assert(!HasPartialSortR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasPartialSortR<UncheckedRange<int*, SentinelForNotWeaklyEqualityComparableWith>>);
static_assert(!HasPartialSortR<UncheckedRange<int*>, BadComparator>);
static_assert(!HasPartialSortR<UncheckedRange<const int*>>); // Doesn't satisfy `sortable`.

template <class Iter, class Sent, std::size_t N>
constexpr void test_one(std::array<int, N> input, std::size_t mid_index, std::array<int, N> sorted) {
  { // (iterator, sentinel) overload.
    auto partially_sorted = input;
    auto begin = Iter(partially_sorted.data());
    auto mid = begin + mid_index;
    auto end = Sent(Iter(partially_sorted.data() + partially_sorted.size()));

    std::same_as<Iter> decltype(auto) last = std::ranges::partial_sort(begin, mid, end);
    assert(std::equal(partially_sorted.begin(), partially_sorted.begin() + mid_index,
                      sorted.begin(), sorted.begin() + mid_index));
    assert(base(last) == partially_sorted.data() + partially_sorted.size());
  }

  { // (range) overload.
    auto partially_sorted = input;
    auto begin = Iter(partially_sorted.data());
    auto mid = begin + mid_index;
    auto end = Sent(Iter(partially_sorted.data() + partially_sorted.size()));
    auto range = std::ranges::subrange(begin, end);

    std::same_as<Iter> decltype(auto) last = std::ranges::partial_sort(range, mid);
    assert(std::ranges::equal(begin, begin + mid_index, sorted.begin(), sorted.begin() + mid_index));
    assert(base(last) == partially_sorted.data() + partially_sorted.size());
  }
}

template <class Iter, class Sent, std::size_t N>
constexpr void test_all_subsequences(std::array<int, N> input) {
  auto sorted = input;
  std::sort(sorted.begin(), sorted.end());

  for (std::size_t n = 0; n <= N; ++n) {
    test_one<Iter, Sent, N>(input, n, sorted);
  }
}

template <class Iter, class Sent>
constexpr void test_iterator_and_sentinel() {
  // Empty sequence.
  test_one<Iter, Sent, 0>({}, 0, {});

  // 1-element sequence.
  test_all_subsequences<Iter, Sent>(std::array{1});

  // 2-element sequence.
  test_all_subsequences<Iter, Sent>(std::array{2, 1});

  // 3-element sequence.
  test_all_subsequences<Iter, Sent>(std::array{2, 1, 3});

  // Longer sequence.
  test_all_subsequences<Iter, Sent>(std::array{2, 1, 3, 6, 8, 4, 11, 5});

  // Longer sequence with duplicates.
  test_all_subsequences<Iter, Sent>(std::array{2, 1, 3, 6, 2, 8, 6});

  // All elements are the same.
  test_all_subsequences<Iter, Sent>(std::array{1, 1, 1});

  // Already sorted.
  test_all_subsequences<Iter, Sent>(std::array{1, 2, 3, 4, 5});

  // Descending.
  test_all_subsequences<Iter, Sent>(std::array{5, 4, 3, 2, 1});

  // Repeating pattern.
  test_all_subsequences<Iter, Sent>(std::array{1, 2, 1, 2, 1, 2});
}

constexpr void test_iterators() {
  test_iterator_and_sentinel<random_access_iterator<int*>, random_access_iterator<int*>>();
  test_iterator_and_sentinel<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  test_iterator_and_sentinel<contiguous_iterator<int*>, contiguous_iterator<int*>>();
  test_iterator_and_sentinel<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  test_iterator_and_sentinel<int*, int*>();
  test_iterator_and_sentinel<int*, sentinel_wrapper<int*>>();
}

constexpr bool test() {
  test_iterators();

  { // A custom comparator works.
    const std::array orig_in = {1, 2, 3, 4, 5};

    {
      auto in = orig_in;
      auto b = in.begin();
      auto m = b + 2;

      auto last = std::ranges::partial_sort(b, m, in.end(), std::ranges::greater{});
      assert(std::ranges::equal(std::ranges::subrange(b, m), std::array{5, 4}));
      assert(last == in.end());
    }

    {
      auto in = orig_in;
      auto b = in.begin();
      auto m = b + 2;

      auto last = std::ranges::partial_sort(in, m, std::ranges::greater{});
      assert(std::ranges::equal(std::ranges::subrange(b, m), std::array{5, 4}));
      assert(last == in.end());
    }
  }

  { // A custom projection works.
    struct A {
      int a;
      constexpr bool operator==(const A&) const = default;
    };

    const std::array orig_in = {A{2}, A{3}, A{1}};

    {
      auto in = orig_in;
      auto b = in.begin();
      auto m = b + 2;

      auto last = std::ranges::partial_sort(b, m, in.end(), {}, &A::a);
      assert(std::ranges::equal(
          std::ranges::subrange(b, m), std::array{A{1}, A{2}}
      ));
      assert(last == in.end());
    }

    {
      auto in = orig_in;
      auto b = in.begin();
      auto m = b + 2;

      auto last = std::ranges::partial_sort(in, m, {}, &A::a);
      assert(std::ranges::equal(
          std::ranges::subrange(b, m), std::array{A{1}, A{2}}
      ));
      assert(last == in.end());
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

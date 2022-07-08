//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//         class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   constexpr I
//     ranges::sort_heap(I first, S last, Comp comp = {}, Proj proj = {});                   // since C++20
//
// template<random_access_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   constexpr borrowed_iterator_t<R>
//     ranges::sort_heap(R&& r, Comp comp = {}, Proj proj = {});                             // since C++20

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
concept HasSortHeapIt = requires(Iter first, Sent last, Comp comp) { std::ranges::make_heap(first, last, comp); };

static_assert(HasSortHeapIt<int*>);
static_assert(!HasSortHeapIt<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasSortHeapIt<RandomAccessIteratorBadIndex>);
static_assert(!HasSortHeapIt<int*, SentinelForNotSemiregular>);
static_assert(!HasSortHeapIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasSortHeapIt<int*, int*, BadComparator>);
static_assert(!HasSortHeapIt<const int*>); // Doesn't satisfy `sortable`.

template <class Range, class Comp = std::ranges::less>
concept HasSortHeapR = requires(Range range, Comp comp) { std::ranges::make_heap(range, comp); };

static_assert(HasSortHeapR<UncheckedRange<int*>>);
static_assert(!HasSortHeapR<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasSortHeapR<RandomAccessRangeBadIndex>);
static_assert(!HasSortHeapR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasSortHeapR<UncheckedRange<int*, SentinelForNotWeaklyEqualityComparableWith>>);
static_assert(!HasSortHeapR<UncheckedRange<int*>, BadComparator>);
static_assert(!HasSortHeapR<UncheckedRange<const int*>>); // Doesn't satisfy `sortable`.

template <size_t N, class T, class Iter>
constexpr void verify_sorted(const std::array<T, N>& sorted, Iter last, std::array<T, N> expected) {
  assert(sorted == expected);
  assert(base(last) == sorted.data() + sorted.size());
  assert(std::is_sorted(sorted.begin(), sorted.end()));
}

template <class Iter, class Sent, size_t N>
constexpr void test_one(const std::array<int, N> input, std::array<int, N> expected) {
  assert(std::is_heap(input.begin(), input.end()));

  { // (iterator, sentinel) overload.
    auto sorted = input;
    auto b = Iter(sorted.data());
    auto e = Sent(Iter(sorted.data() + sorted.size()));

    std::same_as<Iter> decltype(auto) last = std::ranges::sort_heap(b, e);
    verify_sorted(sorted, last, expected);
  }

  { // (range) overload.
    auto sorted = input;
    auto b = Iter(sorted.data());
    auto e = Sent(Iter(sorted.data() + sorted.size()));
    auto range = std::ranges::subrange(b, e);

    std::same_as<Iter> decltype(auto) last = std::ranges::sort_heap(range);
    verify_sorted(sorted, last, expected);
  }
}

template <class Iter, class Sent>
constexpr void test_iterators_2() {
  // 1-element sequence.
  test_one<Iter, Sent, 1>({1}, {1});
  // 2-element sequence.
  test_one<Iter, Sent, 2>({2, 1}, {1, 2});
  // 3-element sequence.
  test_one<Iter, Sent, 3>({3, 1, 2}, {1, 2, 3});
  // Longer sequence.
  test_one<Iter, Sent, 8>({11, 8, 4, 6, 1, 2, 3, 5}, {1, 2, 3, 4, 5, 6, 8, 11});
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 7>({8, 6, 6, 1, 2, 3, 2}, {1, 2, 2, 3, 6, 6, 8});
  // All elements are the same.
  test_one<Iter, Sent, 4>({1, 1, 1, 1}, {1, 1, 1, 1});
}

template <class Iter>
constexpr void test_iterators_1() {
  test_iterators_2<Iter, Iter>();
  test_iterators_2<Iter, sentinel_wrapper<Iter>>();
}

constexpr void test_iterators() {
  test_iterators_1<random_access_iterator<int*>>();
  test_iterators_1<contiguous_iterator<int*>>();
  test_iterators_1<int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom comparator works.
    const std::array input = {1, 2, 3, 5, 4};
    std::array expected = {5, 4, 3, 2, 1};
    auto comp = std::ranges::greater{};
    assert(std::is_heap(input.begin(), input.end(), comp));

    {
      auto in = input;
      auto last = std::ranges::sort_heap(in.begin(), in.end(), comp);
      assert(in == expected);
      assert(last == in.data() + in.size());
    }

    {
      auto in = input;
      auto last = std::ranges::sort_heap(in, comp);
      assert(in == expected);
      assert(last == in.data() + in.size());
    }
  }

  { // A custom projection works.
    struct A {
      int a;
      constexpr auto operator<=>(const A&) const = default;
    };

    const std::array input = {A{3}, A{1}, A{2}};
    std::array expected = {A{1}, A{2}, A{3}};
    {
      auto in = input;
      auto last = std::ranges::sort_heap(in.begin(), in.end(), {}, &A::a);
      verify_sorted(in, last, expected);
    }

    {
      auto in = input;
      auto last = std::ranges::sort_heap(in, {}, &A::a);
      verify_sorted(in, last, expected);
    }
  }

  { // `std::invoke` is used in the implementation.
    struct A {
      int i;
      constexpr A(int i_) : i(i_) {}

      constexpr bool comparator(const A& rhs) const { return i < rhs.i; }
      constexpr const A& projection() const { return *this; }

      constexpr auto operator<=>(const A&) const = default;
    };

    const std::array input = {A{3}, A{1}, A{2}};
    std::array expected = {A{1}, A{2}, A{3}};
    {
      auto in = input;
      auto last = std::ranges::sort_heap(in.begin(), in.end(), &A::comparator, &A::projection);
      verify_sorted(in, last, expected);
    }

    {
      auto in = input;
      auto last = std::ranges::sort_heap(in, &A::comparator, &A::projection);
      verify_sorted(in, last, expected);
    }
  }

  { // The comparator can return any type that's convertible to `bool`.
    const std::array input = {3, 1, 2};
    std::array expected = {1, 2, 3};
    {
      auto in = input;
      auto last = std::ranges::sort_heap(in.begin(), in.end(), [](int i, int j) { return BooleanTestable{i < j}; });
      verify_sorted(in, last, expected);
    }

    {
      auto in = input;
      auto last = std::ranges::sort_heap(in, [](int i, int j) { return BooleanTestable{i < j}; });
      verify_sorted(in, last, expected);
    }
  }

  { // `std::ranges::dangling` is returned.
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::sort_heap(std::array{2, 1, 3});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

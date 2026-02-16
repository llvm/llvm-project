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
//         class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   constexpr I
//     ranges::nth_element(I first, I nth, S last, Comp comp = {}, Proj proj = {});            // since C++20
//
// template<random_access_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   constexpr borrowed_iterator_t<R>
//     ranges::nth_element(R&& r, iterator_t<R> nth, Comp comp = {}, Proj proj = {});          // since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

// SFINAE tests.

using BadComparator = ComparatorNotCopyable<int*>;
static_assert(!std::sortable<int*, BadComparator>);

template <class Iter, class Sent = sentinel_wrapper<Iter>, class Comp = std::ranges::less>
concept HasNthElementIt = requires(Iter first, Iter nth, Sent last, Comp comp) {
  std::ranges::nth_element(first, nth, last, comp);
};

static_assert(HasNthElementIt<int*>);
static_assert(!HasNthElementIt<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasNthElementIt<RandomAccessIteratorBadIndex>);
static_assert(!HasNthElementIt<int*, SentinelForNotSemiregular>);
static_assert(!HasNthElementIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasNthElementIt<int*, int*, BadComparator>);
static_assert(!HasNthElementIt<const int*>); // Doesn't satisfy `sortable`.

template <class Range, class Comp = std::ranges::less>
concept HasNthElementR = requires(Range range, std::ranges::iterator_t<Range> nth, Comp comp) {
  std::ranges::nth_element(range, nth, comp);
};

static_assert(HasNthElementR<UncheckedRange<int*>>);
static_assert(!HasNthElementR<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasNthElementR<RandomAccessRangeBadIndex>);
static_assert(!HasNthElementR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasNthElementR<UncheckedRange<int*, SentinelForNotWeaklyEqualityComparableWith>>);
static_assert(!HasNthElementR<UncheckedRange<int*>, BadComparator>);
static_assert(!HasNthElementR<UncheckedRange<const int*>>); // Doesn't satisfy `sortable`.

template <std::size_t N, class T, class Iter>
constexpr void verify_nth(const std::array<T, N>& partially_sorted, std::size_t nth_index, Iter last, T expected_nth) {
  // Note that the exact output of `nth_element` is unspecified and may vary between implementations.

  assert(base(last) == partially_sorted.data() + partially_sorted.size());

  auto b = partially_sorted.begin();
  auto nth = b + nth_index;
  auto e = partially_sorted.end();
  if (nth == e)
    return;

  assert(*nth == expected_nth);

  // All elements on the left are <= nth.
  assert(std::all_of(b, nth, [&](const auto& v) { return v <= *nth; }));
  // All elements on the right are >= nth.
  assert(std::all_of(nth, e, [&](const auto& v) { return v >= *nth; }));

  {
    auto sorted = partially_sorted;
    std::ranges::sort(sorted);

    // The element at index `n` is the same as if the range were fully sorted.
    assert(sorted[nth_index] == *nth);
  }
}

template <class Iter, class Sent, std::size_t N>
constexpr void test_one(std::array<int, N> input, std::size_t nth_index, std::optional<int> expected_nth = {}) {
  assert(expected_nth || nth_index == N);

  { // (iterator, sentinel) overload.
    auto partially_sorted = input;
    auto b = Iter(partially_sorted.data());
    auto nth = b + nth_index;
    auto e = Sent(Iter(partially_sorted.data() + partially_sorted.size()));

    std::same_as<Iter> decltype(auto) last = std::ranges::nth_element(b, nth, e);
    if (nth_index != N) {
      verify_nth(partially_sorted, nth_index, last, *expected_nth);
    } else {
      assert(partially_sorted == input);
    }
  }

  { // (range) overload.
    auto partially_sorted = input;
    auto b = Iter(partially_sorted.data());
    auto nth = b + nth_index;
    auto e = Sent(Iter(partially_sorted.data() + partially_sorted.size()));
    auto range = std::ranges::subrange(b, e);

    std::same_as<Iter> decltype(auto) last = std::ranges::nth_element(range, nth);
    if (nth_index != N) {
      verify_nth(partially_sorted, nth_index, last, *expected_nth);
    } else {
      assert(partially_sorted == input);
    }
  }
}

template <class Iter, class Sent, std::size_t N>
constexpr void test_all_cases(std::array<int, N> input) {
  auto sorted = input;
  std::sort(sorted.begin(), sorted.end());

  for (int n = 0; n != N; ++n) {
    test_one<Iter, Sent, N>(input, n, sorted[n]);
  }
  test_one<Iter, Sent, N>(input, N);
}

constexpr void test_iterators() {
  auto check = []<class Iter, class Sent> {
    // Empty sequence.
    test_one<Iter, Sent, 0>({}, 0);

    // 1-element sequence.
    test_all_cases<Iter, Sent>(std::array{1});

    // 2-element sequence.
    test_all_cases<Iter, Sent>(std::array{2, 1});

    // 3-element sequence.
    test_all_cases<Iter, Sent>(std::array{2, 1, 3});

    // Longer sequence.
    test_all_cases<Iter, Sent>(std::array{2, 1, 3, 6, 8, 4, 11, 5});

    // Longer sequence with duplicates.
    test_all_cases<Iter, Sent>(std::array{2, 1, 3, 6, 2, 8, 6});

    // All elements are the same.
    test_all_cases<Iter, Sent>(std::array{1, 1, 1, 1});

    { // nth element is in the right place.
      std::array input = {6, 5, 3, 1, 4, 2};
      constexpr std::size_t N = input.size();
      test_one<Iter, Sent, N>(input, 2, /*expected_nth=*/3);
    }

    // Already sorted.
    test_all_cases<Iter, Sent>(std::array{1, 2, 3, 4, 5, 6});

    // Descending.
    test_all_cases<Iter, Sent>(std::array{6, 5, 4, 3, 2, 1});

    // Repeating pattern.
    test_all_cases<Iter, Sent>(std::array{2, 1, 2, 1, 2, 1});
  };

  check.operator()<random_access_iterator<int*>, random_access_iterator<int*>>();
  check.operator()<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  check.operator()<contiguous_iterator<int*>, contiguous_iterator<int*>>();
  check.operator()<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  check.operator()<int*, int*>();
  check.operator()<int*, sentinel_wrapper<int*>>();
}

constexpr bool test() {
  test_iterators();

  { // A custom comparator works.
    const std::array input = {1, 2, 3, 4, 5};
    std::ranges::greater comp;

    {
      auto in = input;
      auto last = std::ranges::nth_element(in.begin(), in.begin() + 1, in.end(), comp);
      assert(in[1] == 4);
      assert(last == in.end());
    }

    {
      auto in = input;
      auto last = std::ranges::nth_element(in, in.begin() + 1, comp);
      assert(in[1] == 4);
      assert(last == in.end());
    }
  }

  { // A custom projection works.
    struct A {
      int a;
      constexpr bool operator==(const A&) const = default;
    };

    const std::array input = {A{2}, A{1}, A{3}};

    {
      auto in = input;
      auto last = std::ranges::nth_element(in.begin(), in.begin() + 1, in.end(), {}, &A::a);
      assert(in[1] == A{2});
      assert(last == in.end());
    }

    {
      auto in = input;
      auto last = std::ranges::nth_element(in, in.begin() + 1, {}, &A::a);
      assert(in[1] == A{2});
      assert(last == in.end());
    }
  }

  { // `std::invoke` is used in the implementation.
    struct S {
      int i;
      constexpr S(int i_) : i(i_) {}

      constexpr bool comparator(const S& rhs) const { return i < rhs.i; }
      constexpr const S& projection() const { return *this; }

      constexpr bool operator==(const S&) const = default;
    };

    const std::array input = {S{2}, S{1}, S{3}};

    {
      auto in = input;
      auto last = std::ranges::nth_element(in.begin(), in.begin() + 1, in.end(), &S::comparator, &S::projection);
      assert(in[1] == S{2});
      assert(last == in.end());
    }

    {
      auto in = input;
      auto last = std::ranges::nth_element(in, in.begin() + 1, &S::comparator, &S::projection);
      assert(in[1] == S{2});
      assert(last == in.end());
    }
  }

  { // `std::ranges::dangling` is returned.
    std::array in{1, 2, 3};
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::nth_element(std::move(in), in.begin());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

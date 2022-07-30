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

// template<input_iterator I1, sentinel_for<I1> S1,
//          random_access_iterator I2, sentinel_for<I2> S2,
//          class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_copyable<I1, I2> && sortable<I2, Comp, Proj2> &&
//            indirect_strict_weak_order<Comp, projected<I1, Proj1>, projected<I2, Proj2>>
//   constexpr partial_sort_copy_result<I1, I2>
//     partial_sort_copy(I1 first, S1 last, I2 result_first, S2 result_last,
//                       Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});               // Since C++20
//
// template<input_range R1, random_access_range R2, class Comp = ranges::less,
//          class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_copyable<iterator_t<R1>, iterator_t<R2>> &&
//            sortable<iterator_t<R2>, Comp, Proj2> &&
//            indirect_strict_weak_order<Comp, projected<iterator_t<R1>, Proj1>,
//                                       projected<iterator_t<R2>, Proj2>>
//   constexpr partial_sort_copy_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>>
//     partial_sort_copy(R1&& r, R2&& result_r, Comp comp = {},
//                       Proj1 proj1 = {}, Proj2 proj2 = {});                               // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "MoveOnly.h"
#include "almost_satisfies_types.h"
#include "test_iterators.h"

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter1 = int*, class Sent1 = int*, class Iter2 = int*, class Sent2 = int*,
          class Comp = std::ranges::less>
concept HasPartialSortCopyIter =
    requires(Iter1&& first1, Sent1&& last1, Iter2&& first2, Sent2&& last2, Comp&& comp) {
      std::ranges::partial_sort_copy(std::forward<Iter1>(first1), std::forward<Sent1>(last1),
          std::forward<Iter2>(first2), std::forward<Sent2>(last2), std::forward<Comp>(comp));
    };

static_assert(HasPartialSortCopyIter<int*, int*, int*, int*, std::ranges::less>);

// !input_iterator<I1>
static_assert(!HasPartialSortCopyIter<InputIteratorNotDerivedFrom>);
static_assert(!HasPartialSortCopyIter<InputIteratorNotIndirectlyReadable>);
static_assert(!HasPartialSortCopyIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S1, I1>
static_assert(!HasPartialSortCopyIter<int*, SentinelForNotSemiregular>);
static_assert(!HasPartialSortCopyIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !random_access_iterator<I2>
static_assert(!HasPartialSortCopyIter<int*, int*, RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasPartialSortCopyIter<int*, int*, RandomAccessIteratorBadIndex>);

// !sentinel_for<S2, I2>
static_assert(!HasPartialSortCopyIter<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasPartialSortCopyIter<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasPartialSortCopyIter<int*, int*, int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartialSortCopyIter<int*, int*, int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// !indirectly_copyable<I1, I2>
static_assert(!HasPartialSortCopyIter<int*, int*, MoveOnly*>);

// !sortable<I2, Comp, Proj2>
static_assert(!HasPartialSortCopyIter<int*, int*, const int*, const int*>);

struct NoComparator {};
// !indirect_strict_weak_order<Comp, projected<I1, Proj1>, projected<I2, Proj2>>
static_assert(!HasPartialSortCopyIter<NoComparator*, NoComparator*, NoComparator*, NoComparator*>);

// Test constraints of the (range) overload.
// ======================================================

template <class T>
using R = UncheckedRange<T>;

template <class Range1 = R<int*>, class Range2 = R<int*>, class Comp = std::ranges::less>
concept HasPartialSortCopyRange =
    requires(Range1&& range1, Range2&& range2, Comp&& comp) {
      std::ranges::partial_sort_copy(std::forward<Range1>(range1), std::forward<Range2>(range2),
          std::forward<Comp>(comp));
    };

static_assert(HasPartialSortCopyRange<R<int*>, R<int*>, std::ranges::less>);

// !input_range<R1>
static_assert(!HasPartialSortCopyRange<InputRangeNotDerivedFrom>);
static_assert(!HasPartialSortCopyRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasPartialSortCopyRange<InputRangeNotInputOrOutputIterator>);

// !random_access_range<R2>
static_assert(!HasPartialSortCopyRange<R<int*>, RandomAccessRangeNotDerivedFrom>);
static_assert(!HasPartialSortCopyRange<R<int*>, RandomAccessRangeBadIndex>);

// !indirectly_copyable<iterator_t<R1>, iterator_t<R2>>
static_assert(!HasPartialSortCopyRange<R<int*>, R<MoveOnly*>>);

// !sortable<iterator_t<R2>, Comp, Proj2>
static_assert(!HasPartialSortCopyRange<R<int*>, R<const int*>>);

// !indirect_strict_weak_order<Comp, projected<iterator_t<R1>, Proj1>, projected<iterator_t<R2>, Proj2>>
static_assert(!HasPartialSortCopyRange<R<NoComparator*>, R<NoComparator*>>);

static_assert(std::is_same_v<std::ranges::partial_sort_copy_result<int, int>, std::ranges::in_out_result<int, int>>);

template <class Iter, class Sent, class OutIter, class OutSent, size_t N>
constexpr void test_one(
    std::array<int, N> input, size_t input_size, size_t output_size, std::array<int, N> sorted) {
  assert(input_size <= N);
  assert(output_size <= N + 1); // To support testing the case where output size exceeds input size.

  using ResultT = std::ranges::partial_sort_copy_result<Iter, OutIter>;
  // To support testing the case where output size exceeds input size; also makes sure calling `out.data() + int()` is
  // valid.
  constexpr size_t OutputSize = N + 1;
  size_t result_size = std::ranges::min(input_size, output_size);

  auto begin = input.data();
  auto end = input.data() + input_size;

  { // (iterator, sentinel) overload.
    std::array<int, OutputSize> out;
    auto out_begin = out.data();
    auto out_end = out.data() + output_size;

    std::same_as<ResultT> decltype(auto) result = std::ranges::partial_sort_copy(
        Iter(begin), Sent(Iter(end)), OutIter(out_begin), OutSent(OutIter(out_end)));

    assert(base(result.in) == input.data() + input_size);
    assert(base(result.out) == out.data() + result_size);

    assert(std::ranges::equal(std::ranges::subrange(out.begin(), out.begin() + result_size),
           std::ranges::subrange(sorted.begin(), sorted.begin() + result_size)));
  }

  { // (range) overload.
    std::array<int, OutputSize> out;
    auto out_begin = out.data();
    auto out_end = out.data() + output_size;
    auto in_range = std::ranges::subrange(Iter(begin), Sent(Iter(end)));
    auto out_range = std::ranges::subrange(OutIter(out_begin), OutSent(OutIter(out_end)));

    std::same_as<ResultT> decltype(auto) result = std::ranges::partial_sort_copy(in_range, out_range);

    assert(base(result.in) == input.data() + input_size);
    assert(base(result.out) == out.data() + result_size);

    assert(std::ranges::equal(std::ranges::subrange(out.begin(), out.begin() + result_size),
           std::ranges::subrange(sorted.begin(), sorted.begin() + result_size)));
  }

}

template <class Iter, class Sent, class OutIter, class OutSent, size_t N>
constexpr void test_all_subsequences(const std::array<int, N> input) {
  auto sorted = input;
  std::sort(sorted.begin(), sorted.end());

  // Whole input, increasing output size. Also check the case when `output_size` exceeds input size.
  for (size_t out_size = 0; out_size <= N + 1; ++out_size) {
    test_one<Iter, Sent, OutIter, OutSent>(input, N, out_size, sorted);
  }
}

template <class InIter, class Sent1, class OutIter, class Sent2>
constexpr void test_iterators_in_sent1_out_sent2() {
  // Empty sequence.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2, 0>({});

  // 1-element sequence.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{1});

  // 2-element sequence.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{2, 1});

  // 3-element sequence.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{2, 1, 3});

  // Longer sequence.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{2, 1, 3, 6, 8, 4, 11, 5});

  // Longer sequence with duplicates.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{2, 1, 3, 6, 2, 8, 6});

  // All elements are the same.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{1, 1, 1});

  // Already sorted.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{1, 2, 3, 4, 5});

  // Descending.
  test_all_subsequences<InIter, Sent1, OutIter, Sent2>(std::array{5, 4, 3, 2, 1});
}

template <class InIter, class Sent1, class OutIter>
constexpr void test_iterators_in_sent1_out() {
  test_iterators_in_sent1_out_sent2<InIter, Sent1, OutIter, OutIter>();
  test_iterators_in_sent1_out_sent2<InIter, Sent1, OutIter, sentinel_wrapper<OutIter>>();
}

template <class InIter, class Sent1>
constexpr void test_iterators_in_sent1() {
  test_iterators_in_sent1_out<InIter, Sent1, random_access_iterator<int*>>();
}

template <class InIter>
constexpr void test_iterators_in() {
  if constexpr (std::sentinel_for<InIter, InIter>) {
    test_iterators_in_sent1<InIter, InIter>();
  }
  test_iterators_in_sent1<InIter, sentinel_wrapper<InIter>>();
}

constexpr void test_iterators() {
  test_iterators_in<cpp20_input_iterator<int*>>();
  // Omitting other iterator types to reduce the combinatorial explosion.
  test_iterators_in<random_access_iterator<int*>>();
  test_iterators_in<const int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom comparator works.
    const std::array in = {1, 2, 3, 4, 5};
    std::ranges::greater comp;

    {
      std::array<int, 2> out;

      auto result = std::ranges::partial_sort_copy(in.begin(), in.end(), out.begin(), out.end(), comp);
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{5, 4}));
    }

    {
      std::array<int, 2> out;

      auto result = std::ranges::partial_sort_copy(in, out, comp);
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{5, 4}));
    }
  }

  { // A custom projection works.
    struct A {
      int a;

      constexpr A() = default;
      constexpr A(int value) : a(value) {}
      constexpr auto operator<=>(const A&) const = default;
    };

    const std::array in = {2, 1, 3};
    auto proj1 = [](int value) { return value * -1; };
    auto proj2 = [](A value) { return value.a * -1; };
    // The projections negate the argument, so the array will appear to be sorted in descending order: [3, 2, 1]
    // (the projections make it appear to the algorithm as [-3, -2, -1]).

    {
      std::array<A, 2> out;

      // No projections: ascending order.
      auto result = std::ranges::partial_sort_copy(in.begin(), in.end(), out.begin(), out.end(), {});
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{1, 2}));
      // Using projections: descending order.
      result = std::ranges::partial_sort_copy(in.begin(), in.end(), out.begin(), out.end(), {}, proj1, proj2);
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{3, 2}));
    }

    {
      std::array<int, 2> out;

      auto result = std::ranges::partial_sort_copy(in, out, {});
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{1, 2}));
      result = std::ranges::partial_sort_copy(in, out, {}, proj1, proj2);
      assert(std::ranges::equal(std::ranges::subrange(out.begin(), result.out), std::array{3, 2}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

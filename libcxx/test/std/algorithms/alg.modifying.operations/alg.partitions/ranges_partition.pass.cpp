//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<permutable I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr subrange<I>
//     partition(I first, S last, Pred pred, Proj proj = {});                                       // Since C++20
//
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R>
//     partition(R&& r, Pred pred, Proj proj = {});                                                 // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct UnaryPred { bool operator()(int) const; };

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Pred = UnaryPred>
concept HasPartitionIter =
    requires(Iter&& iter, Sent&& sent, Pred&& pred) {
      std::ranges::partition(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Pred>(pred));
    };

static_assert(HasPartitionIter<int*, int*, UnaryPred>);

// !permutable<I>
static_assert(!HasPartitionIter<PermutableNotForwardIterator>);
static_assert(!HasPartitionIter<PermutableNotSwappable>);

// !sentinel_for<S, I>
static_assert(!HasPartitionIter<int*, SentinelForNotSemiregular>);
static_assert(!HasPartitionIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasPartitionIter<int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionIter<int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Pred>
concept HasPartitionRange =
    requires(Range&& range, Pred&& pred) {
      std::ranges::partition(std::forward<Range>(range), std::forward<Pred>(pred));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasPartitionRange<R<int*>, UnaryPred>);

// !forward_range<R>
static_assert(!HasPartitionRange<ForwardRangeNotDerivedFrom, UnaryPred>);
static_assert(!HasPartitionRange<ForwardRangeNotIncrementable, UnaryPred>);

// !indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
static_assert(!HasPartitionRange<R<int*>, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionRange<R<int*>, IndirectUnaryPredicateNotCopyConstructible>);

// !permutable<iterator_t<R>>
static_assert(!HasPartitionRange<R<PermutableNotForwardIterator>, UnaryPred>);
static_assert(!HasPartitionRange<R<PermutableNotSwappable>, UnaryPred>);

// `partition` isn't a stable algorithm so this function cannot test the exact output.
template <class Iter, class Sent, std::size_t N, class Pred>
constexpr void test_one(std::array<int, N> input, Pred pred, std::size_t partition_point) {
  auto neg_pred = [&](int x) { return !pred(x); };

  { // (iterator, sentinel) overload.
    auto partitioned = input;
    auto b = Iter(partitioned.data());
    auto e = Sent(Iter(partitioned.data() + partitioned.size()));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::partition(b, e, pred);

    assert(base(result.begin()) == partitioned.data() + partition_point);
    assert(base(result.end()) == partitioned.data() + partitioned.size());

    assert(std::ranges::all_of(b, result.begin(), pred));
    assert(std::ranges::all_of(result.begin(), e, neg_pred));
  }

  { // (range) overload.
    auto partitioned = input;
    auto b = Iter(partitioned.data());
    auto e = Sent(Iter(partitioned.data() + partitioned.size()));
    auto range = std::ranges::subrange(b, e);

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::partition(range, pred);

    assert(base(result.begin()) == partitioned.data() + partition_point);
    assert(base(result.end()) == partitioned.data() + partitioned.size());

    assert(std::ranges::all_of(b, result.begin(), pred));
    assert(std::ranges::all_of(result.begin(), e, neg_pred));
  }
}

template <class Iter, class Sent>
constexpr void test_iterators_2() {
  auto is_odd = [](int x) { return x % 2 != 0; };

  // Empty sequence.
  test_one<Iter, Sent, 0>({}, is_odd, 0);
  // 1-element sequence, the element satisfies the predicate.
  test_one<Iter, Sent, 1>({1}, is_odd, 1);
  // 1-element sequence, the element doesn't satisfy the predicate.
  test_one<Iter, Sent, 1>({2}, is_odd, 0);
  // 2-element sequence, not in order.
  test_one<Iter, Sent, 2>({2, 1}, is_odd, 1);
  // 2-element sequence, already in order.
  test_one<Iter, Sent, 2>({1, 2}, is_odd, 1);
  // 3-element sequence.
  test_one<Iter, Sent, 3>({2, 1, 3}, is_odd, 2);
  // Longer sequence.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 8, 4, 11, 5}, is_odd, 4);
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 2, 8, 1, 6}, is_odd, 3);
  // All elements are the same and satisfy the predicate.
  test_one<Iter, Sent, 3>({1, 1, 1}, is_odd, 3);
  // All elements are the same and don't satisfy the predicate.
  test_one<Iter, Sent, 3>({2, 2, 2}, is_odd, 0);
  // Already partitioned.
  test_one<Iter, Sent, 6>({1, 3, 5, 4, 6, 8}, is_odd, 3);
  // Reverse-partitioned.
  test_one<Iter, Sent, 6>({4, 6, 8, 1, 3, 5}, is_odd, 3);
  // Repeating pattern.
  test_one<Iter, Sent, 6>({1, 2, 1, 2, 1, 2}, is_odd, 3);

  auto is_negative = [](int x) { return x < 0; };
  // Different comparator.
  test_one<Iter, Sent, 5>({-3, 5, 7, -6, 2}, is_negative, 2);
}

template <class Iter>
constexpr void test_iterators_1() {
  test_iterators_2<Iter, Iter>();
  test_iterators_2<Iter, sentinel_wrapper<Iter>>();
}

constexpr void test_iterators() {
  test_iterators_1<forward_iterator<int*>>();
  test_iterators_1<bidirectional_iterator<int*>>();
  test_iterators_1<random_access_iterator<int*>>();
  test_iterators_1<contiguous_iterator<int*>>();
  test_iterators_1<int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom projection works.
    const std::array input = {1, -1};
    auto is_negative = [](int x) { return x < 0; };
    auto negate = [](int x) { return -x; };
    const std::array expected_no_proj = {-1, 1};
    const std::array expected_with_proj = {1, -1};

    { // (iterator, sentinel) overload.
      {
        auto in = input;
        std::ranges::partition(in.begin(), in.end(), is_negative);
        assert(in == expected_no_proj);
      }
      {
        auto in = input;
        std::ranges::partition(in.begin(), in.end(), is_negative, negate);
        assert(in == expected_with_proj);
      }
    }

    { // (range) overload.
      {
        auto in = input;
        std::ranges::partition(in, is_negative);
        assert(in == expected_no_proj);
      }
      {
        auto in = input;
        std::ranges::partition(in, is_negative, negate);
        assert(in == expected_with_proj);
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

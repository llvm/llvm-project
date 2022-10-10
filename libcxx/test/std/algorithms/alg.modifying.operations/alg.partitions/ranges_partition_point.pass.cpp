//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr I partition_point(I first, S last, Pred pred, Proj proj = {});                       // Since C++20
//
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_iterator_t<R>
//     partition_point(R&& r, Pred pred, Proj proj = {});                                           // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct UnaryPred { bool operator()(int) const; };

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Pred = UnaryPred>
concept HasPartitionPointIter =
    requires(Iter&& iter, Sent&& sent, Pred&& pred) {
      std::ranges::partition_point(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Pred>(pred));
    };

static_assert(HasPartitionPointIter<int*, int*, UnaryPred>);

// !forward_iterator<I>
static_assert(!HasPartitionPointIter<ForwardIteratorNotDerivedFrom>);
static_assert(!HasPartitionPointIter<ForwardIteratorNotIncrementable>);

// !sentinel_for<S, I>
static_assert(!HasPartitionPointIter<int*, SentinelForNotSemiregular>);
static_assert(!HasPartitionPointIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasPartitionPointIter<int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionPointIter<int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Pred>
concept HasPartitionPointRange =
    requires(Range&& range, Pred&& pred) {
      std::ranges::partition_point(std::forward<Range>(range), std::forward<Pred>(pred));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasPartitionPointRange<R<int*>, UnaryPred>);

// !forward_range<R>
static_assert(!HasPartitionPointRange<ForwardRangeNotDerivedFrom, UnaryPred>);
static_assert(!HasPartitionPointRange<ForwardRangeNotIncrementable, UnaryPred>);

// !indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
static_assert(!HasPartitionPointRange<R<int*>, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionPointRange<R<int*>, IndirectUnaryPredicateNotCopyConstructible>);

template <class Iter, class Sent, size_t N, class Pred>
constexpr void test_one(std::array<int, N> input, Pred pred, size_t partition_point) {
  assert(std::ranges::is_partitioned(input, pred));

  auto begin = Iter(input.data());
  auto end = Sent(Iter(input.data() + input.size()));
  auto neg_pred = [&](int x) { return !pred(x); };

  { // (iterator, sentinel) overload.
    std::same_as<Iter> decltype(auto) result = std::ranges::partition_point(begin, end, pred);

    assert(base(result) == input.data() + partition_point);
    assert(std::ranges::all_of(begin, result, pred));
    assert(std::ranges::all_of(result, end, neg_pred));
  }

  { // (range) overload.
    auto range = std::ranges::subrange(begin, end);
    std::same_as<Iter> decltype(auto) result = std::ranges::partition_point(range, pred);

    assert(base(result) == input.data() + partition_point);
    assert(std::ranges::all_of(begin, result, pred));
    assert(std::ranges::all_of(result, end, neg_pred));
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
  // 2-element sequence.
  test_one<Iter, Sent, 2>({1, 2}, is_odd, 1);
  // 3-element sequence.
  test_one<Iter, Sent, 3>({3, 1, 2}, is_odd, 2);
  // Longer sequence.
  test_one<Iter, Sent, 8>({1, 3, 11, 5, 6, 2, 8, 4}, is_odd, 4);
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 8>({1, 3, 3, 4, 6, 2, 8, 2}, is_odd, 3);
  // All elements are the same and satisfy the predicate.
  test_one<Iter, Sent, 3>({1, 1, 1}, is_odd, 3);
  // All elements are the same and don't satisfy the predicate.
  test_one<Iter, Sent, 3>({2, 2, 2}, is_odd, 0);
  // All non-satisfying and all satisfying elements are the same.
  test_one<Iter, Sent, 6>({1, 1, 1, 2, 2, 2}, is_odd, 3);

  auto is_negative = [](int x) { return x < 0; };
  // Different comparator.
  test_one<Iter, Sent, 5>({-3, -6, 5, 7, 2}, is_negative, 2);
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
    const std::array in = {1, 3, 4, 6, 8};
    auto is_odd = [](int x) { return x % 2 != 0; };
    auto x2 = [](int x) { return x * 2; };
    auto expected_no_proj = in.begin() + 2;
    auto expected_with_proj = in.begin();

    { // (iterator, sentinel) overload.
      auto result_no_proj = std::ranges::partition_point(in.begin(), in.end(), is_odd);
      assert(result_no_proj == expected_no_proj);
      auto result_with_proj = std::ranges::partition_point(in.begin(), in.end(), is_odd, x2);
      assert(result_with_proj == expected_with_proj);
    }

    { // (range) overload.
      auto result_no_proj = std::ranges::partition_point(in, is_odd);
      assert(result_no_proj == expected_no_proj);
      auto result_with_proj = std::ranges::partition_point(in, is_odd, x2);
      assert(result_with_proj == expected_with_proj);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

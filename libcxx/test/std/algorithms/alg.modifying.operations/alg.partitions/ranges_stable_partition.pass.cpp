//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<bidirectional_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires permutable<I>
//   subrange<I> stable_partition(I first, S last, Pred pred, Proj proj = {});                      // Since C++20
//
// template<bidirectional_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires permutable<iterator_t<R>>
//   borrowed_subrange_t<R> stable_partition(R&& r, Pred pred, Proj proj = {});                     // Since C++20

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
concept HasStablePartitionIter =
    requires(Iter&& iter, Sent&& sent, Pred&& pred) {
      std::ranges::stable_partition(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Pred>(pred));
    };

static_assert(HasStablePartitionIter<int*, int*, UnaryPred>);

// !bidirectional_iterator<I>
static_assert(!HasStablePartitionIter<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasStablePartitionIter<BidirectionalIteratorNotDecrementable>);

// !sentinel_for<S, I>
static_assert(!HasStablePartitionIter<int*, SentinelForNotSemiregular>);
static_assert(!HasStablePartitionIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasStablePartitionIter<int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasStablePartitionIter<int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// !permutable<I>
static_assert(!HasStablePartitionIter<PermutableNotForwardIterator>);
static_assert(!HasStablePartitionIter<PermutableNotSwappable>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Pred>
concept HasStablePartitionRange =
    requires(Range&& range, Pred&& pred) {
      std::ranges::stable_partition(std::forward<Range>(range), std::forward<Pred>(pred));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasStablePartitionRange<R<int*>, UnaryPred>);

// !bidirectional_range<R>
static_assert(!HasStablePartitionRange<BidirectionalRangeNotDerivedFrom, UnaryPred>);
static_assert(!HasStablePartitionRange<BidirectionalRangeNotDecrementable, UnaryPred>);

// !indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
static_assert(!HasStablePartitionRange<R<int*>, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasStablePartitionRange<R<int*>, IndirectUnaryPredicateNotCopyConstructible>);

// !permutable<iterator_t<R>>
static_assert(!HasStablePartitionRange<R<PermutableNotForwardIterator>, UnaryPred>);
static_assert(!HasStablePartitionRange<R<PermutableNotSwappable>, UnaryPred>);

template <class Iter, class Sent, size_t N, class Pred>
void test_one(std::array<int, N> input, Pred pred, size_t partition_point, std::array<int, N> expected) {
  auto neg_pred = [&](int x) { return !pred(x); };

  { // (iterator, sentinel) overload.
    auto partitioned = input;
    auto b = Iter(partitioned.data());
    auto e = Sent(Iter(partitioned.data() + partitioned.size()));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::stable_partition(b, e, pred);

    assert(partitioned == expected);
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

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::stable_partition(range, pred);

    assert(partitioned == expected);
    assert(base(result.begin()) == partitioned.data() + partition_point);
    assert(base(result.end()) == partitioned.data() + partitioned.size());

    assert(std::ranges::all_of(b, result.begin(), pred));
    assert(std::ranges::all_of(result.begin(), e, neg_pred));
  }
}

template <class Iter, class Sent>
void test_iterators_2() {
  auto is_odd = [](int x) { return x % 2 != 0; };

  // Empty sequence.
  test_one<Iter, Sent, 0>({}, is_odd, 0, {});
  // 1-element sequence, the element satisfies the predicate.
  test_one<Iter, Sent, 1>({1}, is_odd, 1, {1});
  // 1-element sequence, the element doesn't satisfy the predicate.
  test_one<Iter, Sent, 1>({2}, is_odd, 0, {2});
  // 2-element sequence, not in order.
  test_one<Iter, Sent, 2>({2, 1}, is_odd, 1, {1, 2});
  // 2-element sequence, already in order.
  test_one<Iter, Sent, 2>({1, 2}, is_odd, 1, {1, 2});
  // 3-element sequence.
  test_one<Iter, Sent, 3>({2, 1, 3}, is_odd, 2, {1, 3, 2});
  // Longer sequence.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 8, 4, 11, 5}, is_odd, 4, {1, 3, 11, 5, 2, 6, 8, 4});
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 2, 8, 1, 6}, is_odd, 3, {1, 3, 1, 2, 6, 2, 8, 6});
  // All elements are the same and satisfy the predicate.
  test_one<Iter, Sent, 3>({1, 1, 1}, is_odd, 3, {1, 1, 1});
  // All elements are the same and don't satisfy the predicate.
  test_one<Iter, Sent, 3>({2, 2, 2}, is_odd, 0, {2, 2, 2});
  // Already partitioned.
  test_one<Iter, Sent, 6>({1, 3, 5, 4, 6, 8}, is_odd, 3, {1, 3, 5, 4, 6, 8});
  // Reverse-partitioned.
  test_one<Iter, Sent, 6>({4, 6, 8, 1, 3, 5}, is_odd, 3, {1, 3, 5, 4, 6, 8});
  // Repeating pattern.
  test_one<Iter, Sent, 6>({1, 2, 1, 2, 1, 2}, is_odd, 3, {1, 1, 1, 2, 2, 2});

  auto is_negative = [](int x) { return x < 0; };
  // Different comparator.
  test_one<Iter, Sent, 5>({-3, 5, 7, -6, 2}, is_negative, 2, {-3, -6, 5, 7, 2});
}

template <class Iter>
void test_iterators_1() {
  test_iterators_2<Iter, Iter>();
  test_iterators_2<Iter, sentinel_wrapper<Iter>>();
}

void test_iterators() {
  test_iterators_1<bidirectional_iterator<int*>>();
  test_iterators_1<random_access_iterator<int*>>();
  test_iterators_1<contiguous_iterator<int*>>();
  test_iterators_1<int*>();
}

void test() {
  test_iterators();

  { // The algorithm is stable (equivalent elements remain in the same order).
    struct OrderedValue {
      int value;
      double original_order;
      bool operator==(const OrderedValue&) const = default;
    };

    auto is_odd = [](OrderedValue x) { return x.value % 2 != 0; };

    using V = OrderedValue;
    using Array = std::array<V, 20>;
    Array orig_in = {
      V{10, 2.1}, {12, 2.2}, {3, 1.1}, {5, 1.2}, {3, 1.3}, {3, 1.4}, {11, 1.5}, {12, 2.3}, {4, 2.4}, {4, 2.5},
      {4, 2.6}, {1, 1.6}, {6, 2.7}, {3, 1.7}, {10, 2.8}, {8, 2.9}, {12, 2.10}, {1, 1.8}, {1, 1.9}, {5, 1.10}
    };
    Array expected = {
      V{3, 1.1}, {5, 1.2}, {3, 1.3}, {3, 1.4}, {11, 1.5}, {1, 1.6}, {3, 1.7}, {1, 1.8}, {1, 1.9}, {5, 1.10},
      {10, 2.1}, {12, 2.2}, {12, 2.3}, {4, 2.4}, {4, 2.5}, {4, 2.6}, {6, 2.7}, {10, 2.8}, {8, 2.9}, {12, 2.10}
    };

    {
      auto in = orig_in;
      std::ranges::stable_partition(in.begin(), in.end(), is_odd);
      assert(in == expected);
    }

    {
      auto in = orig_in;
      std::ranges::stable_partition(in, is_odd);
      assert(in == expected);
    }
  }

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
}

int main(int, char**) {
  test();
  // Note: `stable_partition` is not `constexpr`.

  return 0;
}

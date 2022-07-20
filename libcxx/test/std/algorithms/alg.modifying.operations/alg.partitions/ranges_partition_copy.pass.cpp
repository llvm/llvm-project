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

// template<input_iterator I, sentinel_for<I> S,
//          weakly_incrementable O1, weakly_incrementable O2,
//          class Proj = identity, indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O1> && indirectly_copyable<I, O2>
//   constexpr partition_copy_result<I, O1, O2>
//     partition_copy(I first, S last, O1 out_true, O2 out_false, Pred pred,
//                    Proj proj = {});                                                              // Since C++20
//
// template<input_range R, weakly_incrementable O1, weakly_incrementable O2,
//          class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O1> &&
//            indirectly_copyable<iterator_t<R>, O2>
//   constexpr partition_copy_result<borrowed_iterator_t<R>, O1, O2>
//     partition_copy(R&& r, O1 out_true, O2 out_false, Pred pred, Proj proj = {});                 // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

struct UnaryPred { bool operator()(int) const; };

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class InIter = int*, class Sent = int*, class Output1 = int*, class Output2 = int*, class Pred = UnaryPred>
concept HasPartitionCopyIter =
    requires(InIter&& input, Sent&& sent, Output1&& output1, Output2&& output2, Pred&& pred) {
      std::ranges::partition_copy(std::forward<InIter>(input), std::forward<Sent>(sent),
          std::forward<Output1>(output1), std::forward<Output2>(output2), std::forward<Pred>(pred));
    };

static_assert(HasPartitionCopyIter<int*, int*, int*, int*, UnaryPred>);

// !input_iterator<I>
static_assert(!HasPartitionCopyIter<InputIteratorNotDerivedFrom>);
static_assert(!HasPartitionCopyIter<InputIteratorNotIndirectlyReadable>);
static_assert(!HasPartitionCopyIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, I>
static_assert(!HasPartitionCopyIter<int*, SentinelForNotSemiregular>);
static_assert(!HasPartitionCopyIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !weakly_incrementable<O1>
static_assert(!HasPartitionCopyIter<int*, int*, WeaklyIncrementableNotMovable>);

// !weakly_incrementable<O2>
static_assert(!HasPartitionCopyIter<int*, int*, int*, WeaklyIncrementableNotMovable>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasPartitionCopyIter<int*, int*, int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionCopyIter<int*, int*, int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

struct Uncopyable {
  Uncopyable(int&&);
  Uncopyable(const int&) = delete;
};
// !indirectly_copyable<I, O1>
static_assert(!HasPartitionCopyIter<int*, int*, Uncopyable*>);
// !indirectly_copyable<I, O2>
static_assert(!HasPartitionCopyIter<int*, int*, int*, Uncopyable*>);

// Test constraints of the (range) overload.
// =========================================

template <class InputRange, class Output1 = int*, class Output2 = int*, class Pred = UnaryPred>
concept HasPartitionCopyRange =
    requires(InputRange&& input, Output1&& output1, Output2&& output2, Pred&& pred) {
      std::ranges::partition_copy(std::forward<InputRange>(input),
          std::forward<Output1>(output1), std::forward<Output2>(output2), std::forward<Pred>(pred));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasPartitionCopyRange<R<int*>, int*, int*, UnaryPred>);

// !input_iterator<I>
static_assert(!HasPartitionCopyRange<InputRangeNotDerivedFrom>);
static_assert(!HasPartitionCopyRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasPartitionCopyRange<InputRangeNotInputOrOutputIterator>);

// !weakly_incrementable<O1>
static_assert(!HasPartitionCopyRange<R<int*>, WeaklyIncrementableNotMovable>);

// !weakly_incrementable<O2>
static_assert(!HasPartitionCopyRange<R<int*>, int*, WeaklyIncrementableNotMovable>);

// !indirect_unary_predicate<projected<I, Proj>>
static_assert(!HasPartitionCopyRange<R<int*>, int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasPartitionCopyRange<R<int*>, int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// !indirectly_copyable<I, O1>
static_assert(!HasPartitionCopyRange<R<int*>, Uncopyable*>);
// !indirectly_copyable<I, O2>
static_assert(!HasPartitionCopyRange<R<int*>, int*, Uncopyable*>);

static_assert(std::is_same_v<std::ranges::partition_copy_result<int, int, int>,
    std::ranges::in_out_out_result<int, int, int>>);

template <class Iter, class Sent, class OutIter1, class OutIter2, size_t N1, size_t N2, size_t N3, class Pred>
constexpr void test_one(std::array<int, N1> input, Pred pred, std::array<int, N2> expected_true,
    std::array<int, N3> expected_false) {
  static_assert(N2 + N3 == N1);
  using ResultT = std::ranges::partition_copy_result<Iter, OutIter1, OutIter2>;

  auto begin = input.data();
  auto end = input.data() + input.size();

  { // (iterator, sentinel) overload.
    std::array<int, N2> out1;
    std::array<int, N3> out2;

    std::same_as<ResultT> decltype(auto) result = std::ranges::partition_copy(
        Iter(begin), Sent(Iter(end)), OutIter1(out1.begin()), OutIter2(out2.begin()), pred);

    assert(base(result.in) == input.data() + input.size());
    assert(base(result.out1) == out1.data() + expected_true.size());
    assert(base(result.out2) == out2.data() + expected_false.size());

    assert(std::ranges::equal(out1, expected_true));
    assert(std::ranges::equal(out2, expected_false));
  }

  { // (range) overload.
    std::ranges::subrange range{Iter(begin), Sent(Iter(end))};
    std::array<int, N2> out1;
    std::array<int, N3> out2;

    std::same_as<ResultT> decltype(auto) result = std::ranges::partition_copy(
        range, OutIter1(out1.begin()), OutIter2(out2.begin()), pred);

    assert(base(result.in) == input.data() + input.size());
    assert(base(result.out1) == out1.data() + expected_true.size());
    assert(base(result.out2) == out2.data() + expected_false.size());

    assert(std::ranges::equal(out1, expected_true));
    assert(std::ranges::equal(out2, expected_false));
  }
}

template <class InIter, class Sent, class Out1, class Out2>
constexpr void test_iterators_in_sent_out1_out2() {
  auto is_odd = [](int x) { return x % 2 != 0; };

  // Empty sequence.
  test_one<InIter, Sent, Out1, Out2, 0, 0, 0>({}, is_odd, {}, {});
  // 1-element sequence, the element satisfies the predicate.
  test_one<InIter, Sent, Out1, Out2, 1, 1, 0>({1}, is_odd, {1}, {});
  // 1-element sequence, the element doesn't satisfy the predicate.
  test_one<InIter, Sent, Out1, Out2, 1, 0, 1>({2}, is_odd, {}, {2});
  // 2-element sequence, not in order.
  test_one<InIter, Sent, Out1, Out2, 2, 1, 1>({2, 1}, is_odd, {1}, {2});
  // 2-element sequence, already in order.
  test_one<InIter, Sent, Out1, Out2, 2, 1, 1>({1, 2}, is_odd, {1}, {2});
  // 3-element sequence.
  test_one<InIter, Sent, Out1, Out2, 3, 2, 1>({2, 1, 3}, is_odd, {1, 3}, {2});
  // Longer sequence.
  test_one<InIter, Sent, Out1, Out2, 8, 4, 4>({2, 1, 3, 6, 8, 4, 11, 5}, is_odd, {1, 3, 11, 5}, {2, 6, 8, 4});
  // Longer sequence with duplicates.
  test_one<InIter, Sent, Out1, Out2, 8, 3, 5>({2, 1, 3, 6, 2, 8, 1, 6}, is_odd, {1, 3, 1}, {2, 6, 2, 8, 6});
  // All elements are the same and satisfy the predicate.
  test_one<InIter, Sent, Out1, Out2, 3, 3, 0>({1, 1, 1}, is_odd, {1, 1, 1}, {});
  // All elements are the same and don't satisfy the predicate.
  test_one<InIter, Sent, Out1, Out2, 3, 0, 3>({2, 2, 2}, is_odd, {}, {2, 2, 2});
  // Already partitioned.
  test_one<InIter, Sent, Out1, Out2, 6, 3, 3>({1, 3, 5, 4, 6, 8}, is_odd, {1, 3, 5}, {4, 6, 8});
  // Reverse-partitioned.
  test_one<InIter, Sent, Out1, Out2, 6, 3, 3>({4, 6, 8, 1, 3, 5}, is_odd, {1, 3, 5}, {4, 6, 8});
  // Repeating pattern.
  test_one<InIter, Sent, Out1, Out2, 6, 3, 3>({1, 2, 1, 2, 1, 2}, is_odd, {1, 1, 1}, {2, 2, 2});

  auto is_negative = [](int x) { return x < 0; };
  // Different comparator.
  test_one<InIter, Sent, Out1, Out2, 5, 2, 3>({-3, 5, 7, -6, 2}, is_negative, {-3, -6}, {5, 7, 2});
}

template <class InIter, class Sent, class Out1>
constexpr void test_iterators_in_sent_out1() {
  test_iterators_in_sent_out1_out2<InIter, Sent, Out1, cpp20_output_iterator<int*>>();
  test_iterators_in_sent_out1_out2<InIter, Sent, Out1, random_access_iterator<int*>>();
  test_iterators_in_sent_out1_out2<InIter, Sent, Out1, int*>();
}

template <class InIter, class Sent>
constexpr void test_iterators_in_sent() {
  test_iterators_in_sent_out1<InIter, Sent, cpp17_output_iterator<int*>>();
  test_iterators_in_sent_out1<InIter, Sent, cpp20_output_iterator<int*>>();
  test_iterators_in_sent_out1<InIter, Sent, random_access_iterator<int*>>();
  test_iterators_in_sent_out1<InIter, Sent, int*>();
}

template <class InIter>
constexpr void test_iterators_in() {
  if constexpr (std::sentinel_for<InIter, InIter>) {
    test_iterators_in_sent<InIter, InIter>();
  }
  test_iterators_in_sent<InIter, sentinel_wrapper<InIter>>();
}

constexpr void test_iterators() {
  // Note: deliberately testing with only the weakest and "strongest" iterator types to minimize combinatorial
  // explosion.
  test_iterators_in<cpp20_input_iterator<int*>>();
  test_iterators_in<int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom projection works.
    const std::array in = {1, 3, 9, -2, -5, -8};
    auto is_negative = [](int x) { return x < 0; };
    auto negate = [](int x) { return -x; };
    const std::array expected_negative = {-2, -5, -8};
    const std::array expected_positive = {1, 3, 9};

    { // (iterator, sentinel) overload.
      {
        std::array<int, 3> out1, out2;
        std::ranges::partition_copy(in.begin(), in.end(), out1.begin(), out2.begin(), is_negative);
        assert(out1 == expected_negative);
        assert(out2 == expected_positive);
      }
      {
        std::array<int, 3> out1, out2;
        std::ranges::partition_copy(in.begin(), in.end(), out1.begin(), out2.begin(), is_negative, negate);
        assert(out1 == expected_positive);
        assert(out2 == expected_negative);
      }
    }

    { // (range) overload.
      {
        std::array<int, 3> out1, out2;
        std::ranges::partition_copy(in, out1.begin(), out2.begin(), is_negative);
        assert(out1 == expected_negative);
        assert(out2 == expected_positive);
      }
      {
        std::array<int, 3> out1, out2;
        std::ranges::partition_copy(in, out1.begin(), out2.begin(), is_negative, negate);
        assert(out1 == expected_positive);
        assert(out2 == expected_negative);
      }
    }
  }

  { // Complexity: Exactly `last - first` applications of `pred` and `proj`.
    {
      const std::array in = {-2, -5, -8, -11, -10, -5, 1, 3, 9, 6, 8, 2, 4, 2};
      auto is_negative = [](int x) { return x < 0; };
      std::array<int, 6> out1;
      std::array<int, 8> out2;

      int pred_count = 0, proj_count = 0;
      counting_predicate pred(is_negative, pred_count);
      counting_projection proj(proj_count);
      auto expected = static_cast<int>(in.size());

      {
        std::ranges::partition_copy(in.begin(), in.end(), out1.begin(), out2.begin(), pred, proj);
        assert(pred_count == expected);
        assert(proj_count == expected);
        pred_count = proj_count = 0;
      }

      {
        std::ranges::partition_copy(in, out1.begin(), out2.begin(), pred, proj);
        assert(pred_count == expected);
        assert(proj_count == expected);
        pred_count = proj_count = 0;
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

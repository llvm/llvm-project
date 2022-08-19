//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<bidirectional_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//          class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   constexpr ranges::prev_permutation_result<I>
//     ranges::prev_permutation(I first, S last, Comp comp = {}, Proj proj = {});
// template<bidirectional_range R, class Comp = ranges::less,
//          class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   constexpr ranges::prev_permutation_result<borrowed_iterator_t<R>>
//     ranges::prev_permutation(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasPrevPermutationIt = requires(Iter first, Sent last) { std::ranges::prev_permutation(first, last); };

static_assert(HasPrevPermutationIt<int*>);
static_assert(!HasPrevPermutationIt<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasPrevPermutationIt<BidirectionalIteratorNotDecrementable>);
static_assert(!HasPrevPermutationIt<int*, SentinelForNotSemiregular>);
static_assert(!HasPrevPermutationIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasPrevPermutationIt<const int*>); // not sortable

template <class Range>
concept HasPrevPermutationR = requires(Range range) { std::ranges::prev_permutation(range); };

static_assert(HasPrevPermutationR<UncheckedRange<int*>>);
static_assert(!HasPrevPermutationR<BidirectionalRangeNotDerivedFrom>);
static_assert(!HasPrevPermutationR<BidirectionalRangeNotDecrementable>);
static_assert(!HasPrevPermutationR<BidirectionalRangeNotSentinelSemiregular>);
static_assert(!HasPrevPermutationR<BidirectionalRangeNotSentinelWeaklyEqualityComparableWith>);
static_assert(!HasPrevPermutationR<UncheckedRange<const int*>>); // not sortable

constexpr size_t factorial(size_t i) {
  std::array memoized = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
  return memoized[i];
}

template <class Iter, class Range, class Func>
constexpr bool run_prev_permutation(Func call_prev_permutation, Range permuted, Range previous) {
  using Result = std::ranges::prev_permutation_result<Iter>;

  std::same_as<Result> decltype(auto) ret = call_prev_permutation(permuted);
  assert(ret.in == permuted.end());
  bool next_found = ret.found;

  if (std::ranges::distance(permuted) > 1) {
    if (next_found) {
      assert(std::ranges::lexicographical_compare(permuted, previous));
    } else {
      assert(std::ranges::lexicographical_compare(previous, permuted));
      assert(std::ranges::is_sorted(permuted, std::ranges::greater{}));
    }
  }

  return next_found;
}

template <class Iter, class Sent, class Func>
constexpr void test_prev_permutations(Func call_prev_permutation) {
  std::array input = {4, 3, 2, 1};
  auto current_permutation = input;
  auto previous_permutation = current_permutation;

  // For all subarrays of `input` from `[0, 0]` to `[0, N - 1]`, call `prev_permutation` until no previous permutation
  // exists.
  // The number of permutations must equal `N!`. `run_prev_permutation` checks that each previous permutation is
  // lexicographically less than the previous. If these two conditions hold (the number of permutations is `N!`, and
  // each permutation is lexicographically less than the previous one), it follows that the `ranges::prev_permutation`
  // algorithm works correctly.
  for (size_t i = 0; i <= current_permutation.size(); ++i) {
    size_t count = 0;
    bool next_found = true;

    while (next_found) {
      ++count;
      previous_permutation = current_permutation;

      auto current_subrange = std::ranges::subrange(
          Iter(current_permutation.data()), Sent(Iter(current_permutation.data() + i)));
      auto previous_subrange = std::ranges::subrange(
          Iter(previous_permutation.data()), Sent(Iter(previous_permutation.data() + i)));

      next_found = run_prev_permutation<Iter>(call_prev_permutation, current_subrange, previous_subrange);
    }

    assert(count == factorial(i));
  }
}

template <class Iter, class Sent>
constexpr void test_all_permutations() {
  test_prev_permutations<Iter, Sent>([](auto&& range) {
    return std::ranges::prev_permutation(range.begin(), range.end());
  });

  test_prev_permutations<Iter, Sent>([](auto&& range) {
    return std::ranges::prev_permutation(range);
  });
}

template <class Iter, class Sent, int N>
constexpr void test_one(const std::array<int, N> input, bool expected_found, std::array<int, N> expected) {
  using Result = std::ranges::prev_permutation_result<Iter>;

  { // (iterator, sentinel) overload.
    auto in = input;
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));

    std::same_as<Result> decltype(auto) result = std::ranges::prev_permutation(begin, end);
    assert(result.found == expected_found);
    assert(result.in == end);
    assert(in == expected);
  }

  {  // (range) overload.
    auto in = input;
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));
    auto range = std::ranges::subrange(begin, end);

    std::same_as<Result> decltype(auto) result = std::ranges::prev_permutation(range);
    assert(result.found == expected_found);
    assert(result.in == end);
    assert(in == expected);
  }
}

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  test_all_permutations<Iter, Sent>();

  // Empty range.
  test_one<Iter, Sent, 0>({}, false, {});
  // 1-element range.
  test_one<Iter, Sent, 1>({1}, false, {1});
  // 2-element range.
  test_one<Iter, Sent, 2>({1, 2}, false, {2, 1});
  test_one<Iter, Sent, 2>({2, 1}, true, {1, 2});
  // Longer sequence.
  test_one<Iter, Sent, 8>({8, 7, 6, 5, 4, 3, 2, 1}, true, {8, 7, 6, 5, 4, 3, 1, 2});
  // Longer sequence, permutations exhausted.
  test_one<Iter, Sent, 8>({1, 2, 3, 4, 5, 6, 7, 8}, false, {8, 7, 6, 5, 4, 3, 2, 1});
}

template <class Iter>
constexpr void test_iter() {
  test_iter_sent<Iter, Iter>();
  test_iter_sent<Iter, sentinel_wrapper<Iter>>();
  test_iter_sent<Iter, sized_sentinel<Iter>>();
}

constexpr void test_iterators() {
  test_iter<bidirectional_iterator<int*>>();
  test_iter<random_access_iterator<int*>>();
  test_iter<contiguous_iterator<int*>>();
  test_iter<int*>();
}

constexpr bool test() {
  test_iterators();

  { // A custom predicate works.
    struct A {
      int i;
      constexpr bool comp(const A& rhs) const { return i > rhs.i; }
      constexpr bool operator==(const A&) const = default;
    };
    const std::array input = {A{5}, A{4}, A{3}, A{2}, A{1}};
    std::array expected = {A{1}, A{2}, A{3}, A{4}, A{5}};

    { // (iterator, sentinel) overload.
      auto in = input;
      auto result = std::ranges::prev_permutation(in.begin(), in.end(), &A::comp);

      assert(result.found == false);
      assert(result.in == in.end());
      assert(in == expected);
    }

    { // (range) overload.
      auto in = input;
      auto result = std::ranges::prev_permutation(in, &A::comp);

      assert(result.found == false);
      assert(result.in == in.end());
      assert(in == expected);
    }
  }

  { // A custom projection works.
    struct A {
      int i;
      constexpr A negate() const { return A{i * -1}; }
      constexpr auto operator<=>(const A&) const = default;
    };
    const std::array input = {A{5}, A{4}, A{3}, A{2}, A{1}};
    std::array expected = {A{1}, A{2}, A{3}, A{4}, A{5}};

    { // (iterator, sentinel) overload.
      auto in = input;
      auto result = std::ranges::prev_permutation(in.begin(), in.end(), {}, &A::negate);

      assert(result.found == false);
      assert(result.in == in.end());
      assert(in == expected);
    }

    { // (range) overload.
      auto in = input;
      auto result = std::ranges::prev_permutation(in, {}, &A::negate);

      assert(result.found == false);
      assert(result.in == in.end());
      assert(in == expected);
    }
  }

  { // Complexity: At most `(last - first) / 2` swaps.
    const std::array input = {6, 5, 4, 3, 2, 1};

    { // (iterator, sentinel) overload.
      auto in = input;
      int swaps_count = 0;
      auto begin = adl::Iterator::TrackSwaps(in.data(), swaps_count);
      auto end = adl::Iterator::TrackSwaps(in.data() + in.size(), swaps_count);

      std::ranges::prev_permutation(begin, end);
      assert(swaps_count <= (base(end) - base(begin) + 1) / 2);
    }

    { // (range) overload.
      auto in = input;
      int swaps_count = 0;
      auto begin = adl::Iterator::TrackSwaps(in.data(), swaps_count);
      auto end = adl::Iterator::TrackSwaps(in.data() + in.size(), swaps_count);

      std::ranges::prev_permutation(std::ranges::subrange(begin, end));
      assert(swaps_count <= (base(end) - base(begin) + 1) / 2);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

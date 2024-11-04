//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_binary_predicate<projected<I, Proj>,
//                                    projected<I, Proj>> Pred = ranges::equal_to>
//   constexpr I ranges::adjacent_find(I first, S last, Pred pred = {}, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_binary_predicate<projected<iterator_t<R>, Proj>,
//                                    projected<iterator_t<R>, Proj>> Pred = ranges::equal_to>
//   constexpr borrowed_iterator_t<R> ranges::adjacent_find(R&& r, Pred pred = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Sent = Iter>
concept HasAdjacentFindIt = requires (Iter iter, Sent sent) { std::ranges::adjacent_find(iter, sent); };

struct NotComparable {};

static_assert(HasAdjacentFindIt<int*>);
static_assert(!HasAdjacentFindIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasAdjacentFindIt<ForwardIteratorNotIncrementable>);
static_assert(!HasAdjacentFindIt<int*, SentinelForNotSemiregular>);
static_assert(!HasAdjacentFindIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasAdjacentFindIt<NotComparable*>);

template <class Range>
concept HasAdjacentFindR = requires (Range range) { std::ranges::adjacent_find(range); };

static_assert(HasAdjacentFindR<UncheckedRange<int*>>);
static_assert(!HasAdjacentFindR<ForwardRangeNotDerivedFrom>);
static_assert(!HasAdjacentFindR<ForwardRangeNotIncrementable>);
static_assert(!HasAdjacentFindR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasAdjacentFindR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasAdjacentFindR<UncheckedRange<NotComparable>>);

template <std::size_t N>
struct Data {
  std::array<int, N> input;
  int expected;
};

template <class Iter, class Sent, std::size_t N>
constexpr void test(Data<N> d) {
  {
    std::same_as<Iter> decltype(auto) ret =
        std::ranges::adjacent_find(Iter(d.input.data()), Sent(Iter(d.input.data() + d.input.size())));
    assert(base(ret) == d.input.data() + d.expected);
  }
  {
    auto range = std::ranges::subrange(Iter(d.input.data()), Sent(Iter(d.input.data() + d.input.size())));
    std::same_as<Iter> decltype(auto) ret = std::ranges::adjacent_find(range);
    assert(base(ret) == d.input.data() + d.expected);
  }
}

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  // simple test
  test<Iter, Sent, 4>({.input = {1, 2, 2, 4}, .expected = 1});
  // last is returned with no match
  test<Iter, Sent, 4>({.input = {1, 2, 3, 4}, .expected = 4});
  // first elements match
  test<Iter, Sent, 4>({.input = {1, 1, 3, 4}, .expected = 0});
  // the first match is returned
  test<Iter, Sent, 7>({.input = {1, 1, 3, 4, 4, 4, 4}, .expected = 0});
  // two element range works
  test<Iter, Sent, 2>({.input = {3, 3}, .expected = 0});
  // single element range works
  test<Iter, Sent, 1>({.input = {1}, .expected = 1});
  // empty range works
  test<Iter, Sent, 0>({.input = {}, .expected = 0});
}

constexpr bool test() {
  test_iterators<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();
  test_iterators<const int*>();

  { // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::adjacent_find(std::array{1, 2, 3, 4});
  }

  { // check that the complexity requirements are met with no match
    {
      int predicateCount = 0;
      auto pred = [&](int, int) { ++predicateCount; return false; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::adjacent_find(a, a + 5, pred, proj);
      assert(ret == a + 5);
      assert(predicateCount == 4);
      assert(projectionCount == 8);
    }
    {
      int predicateCount = 0;
      auto pred = [&](int, int) { ++predicateCount; return false; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::adjacent_find(a, pred, proj);
      assert(ret == a + 5);
      assert(predicateCount == 4);
      assert(projectionCount == 8);
    }
  }

  { // check that the complexity requirements are met with a match
    {
      int predicateCount = 0;
      auto pred = [&](int i, int j) { ++predicateCount; return i == j; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 4, 4, 5};
      auto ret = std::ranges::adjacent_find(a, a + 5, pred, proj);
      assert(ret == a + 2);
      assert(predicateCount == 3);
      assert(projectionCount == 6);
    }
    {
      int predicateCount = 0;
      auto pred = [&](int i, int j) { ++predicateCount; return i == j; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 4, 4, 5};
      auto ret = std::ranges::adjacent_find(a, pred, proj);
      assert(ret == a + 2);
      assert(predicateCount == 3);
      assert(projectionCount == 6);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      constexpr bool compare(const S& j) const { return j.i == i; }
      constexpr const S& identity() const { return *this; }
      int i;
    };
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::adjacent_find(std::begin(a), std::end(a), &S::compare, &S::identity);
      assert(ret == a + 4);
    }
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::adjacent_find(a, &S::compare, &S::identity);
      assert(ret == a + 4);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

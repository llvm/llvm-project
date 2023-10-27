//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<I, Proj>> Comp = ranges::less>
//   constexpr bool ranges::binary_search(I first, S last, const T& value, Comp comp = {},
//                                        Proj proj = {});
// template<forward_range R, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<iterator_t<R>, Proj>> Comp =
//            ranges::less>
//   constexpr bool ranges::binary_search(R&& r, const T& value, Comp comp = {},
//                                        Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotLessThanComparable {};

template <class It, class Sent = It>
concept HasLowerBoundIt = requires(It it, Sent sent) { std::ranges::binary_search(it, sent, 1); };

static_assert(HasLowerBoundIt<int*>);
static_assert(!HasLowerBoundIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);
static_assert(!HasLowerBoundIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasLowerBoundIt<ForwardIteratorNotIncrementable>);
static_assert(!HasLowerBoundIt<NotLessThanComparable*>);

template <class Range>
concept HasLowerBoundR = requires(Range range) { std::ranges::binary_search(range, 1); };

static_assert(HasLowerBoundR<std::array<int, 1>>);
static_assert(!HasLowerBoundR<ForwardRangeNotDerivedFrom>);
static_assert(!HasLowerBoundR<ForwardRangeNotIncrementable>);
static_assert(!HasLowerBoundR<UncheckedRange<NotLessThanComparable*>>);

template <class Pred>
concept HasLowerBoundPred = requires(int* it, Pred pred) {std::ranges::binary_search(it, it, 1, pred); };

static_assert(HasLowerBoundPred<std::ranges::less>);
static_assert(!HasLowerBoundPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasLowerBoundPred<IndirectUnaryPredicateNotPredicate>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      std::same_as<bool> auto ret = std::ranges::binary_search(It(a), Sent(It(a + 6)), 3);
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 6)));
      std::same_as<bool> auto ret = std::ranges::binary_search(range, 3);
      assert(ret);
    }
  }

  { // check that the predicate is used
    int a[] = {6, 5, 4, 3, 2, 1};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 6)), 2, std::ranges::greater{}));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 6)));
    assert(std::ranges::binary_search(range, 2, std::ranges::greater{}));
  }

  { // check that the projection is used
    int a[] = {1, 2, 3, 4, 5, 6};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 6)), 0, {}, [](int i) { return i - 3; }));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 6)));
    assert(std::ranges::binary_search(range, 0, {}, [](int i) { return i - 3; }));
  }

  { // check that true is returned with multiple matches
    int a[] = {1, 2, 2, 2, 3};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 5)), 2));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 5)));
    assert(std::ranges::binary_search(range, 2));
  }

  { // check that false is returned if all elements compare less than
    int a[] = {1, 2, 3, 4};
    assert(!std::ranges::binary_search(It(a), Sent(It(a + 4)), 5));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
    assert(!std::ranges::binary_search(range, 5));
  }

  { // check that false is returned if no element compares less than
    int a[] = {1, 2, 3, 4};
    assert(!std::ranges::binary_search(It(a), Sent(It(a + 4)), 0));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
    assert(!std::ranges::binary_search(range, 0));
  }

  { // check that a single element works
    int a[] = {1};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 1)), 1));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 1)));
    assert(std::ranges::binary_search(range, 1));
  }

  { // check that an even number of elements works and that searching for the first element works
    int a[] = {1, 2, 7, 8, 10, 11};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 6)), 1));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 6)));
    assert(std::ranges::binary_search(range, 1));
  }

  { // check that an odd number of elements works and that searching for the last element works
    int a[] = {1, 2, 7, 10, 11};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 5)), 11));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 5)));
    assert(std::ranges::binary_search(range, 11));
  }

  { // check that it works when all but the searched for elements are equal
    int a[] = {1, 2, 2, 2, 2};
    assert(std::ranges::binary_search(It(a), Sent(It(a + 5)), 1));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 5)));
    assert(std::ranges::binary_search(range, 1));
  }

  { // check that false is returned when the element doesn't exist, but an element with a greater value is in the range
    int a[] = {1, 2, 4};
    assert(!std::ranges::binary_search(It(a), Sent(It(a + 3)), 3));
    auto range = std::ranges::subrange(It(a), Sent(It(a + 3)));
    assert(!std::ranges::binary_search(range, 3));
  }
}

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();

  { // check that std::invoke is used
    struct S { int check; int other; };
    S a[] = {{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
    assert(std::ranges::binary_search(a, a + 6, 4, {}, &S::check));
    assert(std::ranges::binary_search(a, 4, {}, &S::check));
  }

  { // check that an empty range works
    std::array<int, 0> a;
    assert(!std::ranges::binary_search(a.begin(), a.end(), 1));
    assert(!std::ranges::binary_search(a, 1));
  }

  { // check that a non-const operator() works
    struct Func {
      constexpr bool operator()(const int& i, const int& j) { return i < j; }
    };
    int a[] = {1, 6, 9, 10, 23};
    assert(std::ranges::binary_search(a, 6, Func{}));
    assert(std::ranges::binary_search(a, a + 5, 6, Func{}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

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
//   constexpr I ranges::upper_bound(I first, S last, const T& value, Comp comp = {}, Proj proj = {});
// template<forward_range R, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<iterator_t<R>, Proj>> Comp =
//            ranges::less>
//   constexpr borrowed_iterator_t<R>
//     ranges::upper_bound(R&& r, const T& value, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotLessThanComparable {};

template <class It, class Sent = It>
concept HasUpperBoundIt = requires(It it, Sent sent) { std::ranges::upper_bound(it, sent, 1); };

static_assert(HasUpperBoundIt<int*>);
static_assert(!HasUpperBoundIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);
static_assert(!HasUpperBoundIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasUpperBoundIt<ForwardIteratorNotIncrementable>);
static_assert(!HasUpperBoundIt<NotLessThanComparable*>);

template <class Range>
concept HasUpperBoundR = requires(Range range) { std::ranges::upper_bound(range, 1); };

static_assert(HasUpperBoundR<std::array<int, 1>>);
static_assert(!HasUpperBoundR<ForwardRangeNotDerivedFrom>);
static_assert(!HasUpperBoundR<ForwardRangeNotIncrementable>);
static_assert(!HasUpperBoundR<UncheckedRange<NotLessThanComparable*>>);

template <class Pred>
concept HasUpperBoundPred = requires(int* it, Pred pred) {std::ranges::upper_bound(it, it, 1, pred); };

static_assert(HasUpperBoundPred<std::ranges::less>);
static_assert(!HasUpperBoundPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasUpperBoundPred<IndirectUnaryPredicateNotPredicate>);

template <class It>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      std::same_as<It> auto ret = std::ranges::upper_bound(It(a), It(a + 6), 3);
      assert(base(ret) == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      auto range = std::ranges::subrange(It(a), It(a + 6));
      std::same_as<It> auto ret = std::ranges::upper_bound(range, 3);
      assert(base(ret) == a + 3);
    }
  }

  { // check that the predicate is used
    {
      int a[] = {6, 5, 4, 3, 2, 1};
      auto ret = std::ranges::upper_bound(It(a), It(a + 6), 2, std::ranges::greater{});
      assert(base(ret) == a + 5);
    }
    {
      int a[] = {6, 5, 4, 3, 2, 1};
      auto range = std::ranges::subrange(It(a), It(a + 6));
      auto ret = std::ranges::upper_bound(range, 2, std::ranges::greater{});
      assert(base(ret) == a + 5);
    }
  }

  { // check that the projection is used
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      auto ret = std::ranges::upper_bound(It(a), It(a + 6), 0, {}, [](int i) { return i - 3; });
      assert(base(ret) == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      auto range = std::ranges::subrange(It(a), It(a + 6));
      auto ret = std::ranges::upper_bound(range, 0, {}, [](int i) { return i - 3; });
      assert(base(ret) == a + 3);
    }
  }

  { // check that the last upper bound is returned
    {
      int a[] = {1, 2, 2, 2, 3};
      auto ret = std::ranges::upper_bound(It(a), It(a + 5), 2);
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {1, 2, 2, 2, 3};
      auto range = std::ranges::subrange(It(a), It(a + 5));
      auto ret = std::ranges::upper_bound(range, 2);
      assert(base(ret) == a + 4);
    }
  }

  { // check that end is returned if all elements compare less than
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::upper_bound(It(a), It(a + 4), 5);
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), It(a + 4));
      auto ret = std::ranges::upper_bound(range, 5);
      assert(base(ret) == a + 4);
    }
  }

  { // check that the first element is returned if no element compares less than
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::upper_bound(It(a), It(a + 4), 0);
      assert(base(ret) == a);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), It(a + 4));
      auto ret = std::ranges::upper_bound(range, 0);
      assert(base(ret) == a);
    }
  }

  { // check that a single element works
    {
      int a[] = {1};
      auto ret = std::ranges::upper_bound(It(a), It(a + 1), 1);
      assert(base(ret) == a + 1);
    }
    {
      int a[] = {1};
      auto range = std::ranges::subrange(It(a), It(a + 1));
      auto ret = std::ranges::upper_bound(range, 1);
      assert(base(ret) == a + 1);
    }
  }

  { // check that an even number of elements works
    {
      int a[] = {1, 3, 6, 6, 7, 8};
      auto ret = std::ranges::upper_bound(It(a), It(a + 6), 6);
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {1, 3, 6, 6, 7, 8};
      auto range = std::ranges::subrange(It(a), It(a + 6));
      auto ret = std::ranges::upper_bound(range, 6);
      assert(base(ret) == a + 4);
    }
  }

  { // check that an odd number of elements works
    {
      int a[] = {1, 3, 6, 6, 7};
      auto ret = std::ranges::upper_bound(It(a), It(a + 5), 6);
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {1, 3, 6, 6, 7};
      auto range = std::ranges::subrange(It(a), It(a + 5));
      auto ret = std::ranges::upper_bound(range, 6);
      assert(base(ret) == a + 4);
    }
  }

  { // check that it works when all but the searched for element are equal
    {
      int a[] = {1, 6, 6, 6, 6, 6};
      auto ret = std::ranges::upper_bound(It(a), It(a + 6), 1);
      assert(base(ret) == a + 1);
    }
    {
      int a[] = {1, 6, 6, 6, 6, 6};
      auto range = std::ranges::subrange(It(a), It(a + 6));
      auto ret = std::ranges::upper_bound(range, 1);
      assert(base(ret) == a + 1);
    }
  }

  { // check that the middle of a range is returned when there are smaller and larger elements
    {
      int a[] = {1, 2, 3, 4, 6, 7, 8};
      auto ret = std::ranges::upper_bound(It(a), It(a + 7), 5);
      assert(base(ret) == a + 4);
      assert(*ret == 6);
    }
    {
      int a[] = {1, 2, 3, 4, 6, 7, 8};
      auto range = std::ranges::subrange(It(a), It(a + 7));
      auto ret = std::ranges::upper_bound(range, 5);
      assert(base(ret) == a + 4);
      assert(*ret == 6);
    }
  }
}

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  { // check that std::invoke is used for the projections
    {
      struct S { int check; int other; };
      S a[] = {{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
      auto ret = std::ranges::upper_bound(a, a + 6, 4, {}, &S::check);
      assert(ret == a + 4);
    }
    {
      struct S { int check; int other; };
      S a[] = {{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
      auto ret = std::ranges::upper_bound(a, 4, {}, &S::check);
      assert(ret == a + 4);
    }
  }

  { // check that std::invoke is used for the predicate
    struct S {
      int check;
      int other;

      constexpr bool compare(const S& s) const {
        return check < s.check;
      }
    };
    {
      S a[] = {{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
      auto ret = std::ranges::upper_bound(a, a + 6, S{4, 0}, &S::compare);
      assert(ret == a + 4);
    }
    {
      S a[] = {{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
      auto ret = std::ranges::upper_bound(a, S{4, 0}, &S::compare);
      assert(ret == a + 4);
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> a;
      auto ret = std::ranges::upper_bound(a.begin(), a.end(), 1);
      assert(ret == a.end());
    }
    {
      std::array<int, 0> a;
      auto ret = std::ranges::upper_bound(a, 1);
      assert(ret == a.end());
    }
  }

  { // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret = std::ranges::upper_bound(std::array{1, 2}, 1);
  }

  { // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3};
    std::same_as<int*> auto ret = std::ranges::upper_bound(std::views::all(a), 1);
    assert(ret == a + 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

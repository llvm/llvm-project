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

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_strict_weak_order<projected<I, Proj>> Comp = ranges::less>
//   constexpr bool ranges::is_sorted(I first, S last, Comp comp = {}, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   constexpr bool ranges::is_sorted(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Sent = Iter>
concept HasIsSortedIt = requires(Iter iter, Sent sent) { std::ranges::is_sorted(iter, sent); };

struct HasNoComparator {};

static_assert(HasIsSortedIt<int*>);
static_assert(!HasIsSortedIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasIsSortedIt<ForwardIteratorNotIncrementable>);
static_assert(!HasIsSortedIt<int*, SentinelForNotSemiregular>);
static_assert(!HasIsSortedIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasIsSortedIt<HasNoComparator*>);

template <class Range>
concept HasIsSortedR = requires(Range range) { std::ranges::is_sorted(range); };

static_assert(HasIsSortedR<UncheckedRange<int*>>);
static_assert(!HasIsSortedR<ForwardRangeNotDerivedFrom>);
static_assert(!HasIsSortedR<ForwardRangeNotIncrementable>);
static_assert(!HasIsSortedR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasIsSortedR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasIsSortedR<UncheckedRange<HasNoComparator*>>);

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 3};
      std::same_as<bool> decltype(auto) ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a + 5)));
      assert(!ret);
    }
    {
      int a[] = {1, 2, 3, 4, 3};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      std::same_as<bool> decltype(auto) ret = std::ranges::is_sorted(range);
      assert(!ret);
    }
  }

  { // second element isn't sorted
    {
      int a[] = {1, 0, 3, 4, 5};
      auto ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a + 5)));
      assert(!ret);
    }
    {
      int a[] = {1, 0, 3, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::is_sorted(range);
      assert(!ret);
    }
  }

  { // all elements are sorted
    {
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a + 5)));
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::is_sorted(range);
      assert(ret);
    }
  }

  { // check that the comparator is used
    {
      int a[] = {5, 4, 3, 2};
      auto ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a + 4)), std::ranges::greater{});
      assert(ret);
    }
    {
      int a[] = {5, 4, 3, 2};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::is_sorted(range, std::ranges::greater{});
      assert(ret);
    }
  }

  { // check that an empty range works
    {
      int a[] = {};
      auto ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a)));
      assert(ret);
    }
    {
      int a[] = {};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a)));
      auto ret = std::ranges::is_sorted(range);
      assert(ret);
    }
  }

  { // check that a range with a single element works
    {
      int a[] = {32};
      auto ret = std::ranges::is_sorted(Iter(a), Sent(Iter(a + 1)));
      assert(ret);
    }
    {
      int a[] = {32};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 1)));
      auto ret = std::ranges::is_sorted(range);
      assert(ret);
    }
  }
}

constexpr bool test() {
  test_iterators<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();
  test_iterators<const int*>();

  { // check that the projection is used
    struct S {
      int i;
      constexpr S(int i_) : i(i_) {}
    };
    {
      S a[] = {1, 2, 3};
      auto ret = std::ranges::is_sorted(a, a + 3, {}, &S::i);
      assert(ret);
    }
    {
      S a[] = {1, 2, 3};
      auto ret = std::ranges::is_sorted(a, {}, &S::i);
      assert(ret);
    }
  }

  { // check that a dangling range works
    assert(std::ranges::is_sorted(std::array{1, 2, 3, 4}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

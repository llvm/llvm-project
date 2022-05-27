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
//   constexpr I ranges::is_sorted_until(I first, S last, Comp comp = {}, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   constexpr borrowed_iterator_t<R>
//     ranges::is_sorted_until(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Sent = Iter>
concept HasIsSortedUntilIt = requires(Iter iter, Sent sent) { std::ranges::is_sorted_until(iter, sent); };

struct HasNoComparator {};

static_assert(HasIsSortedUntilIt<int*>);
static_assert(!HasIsSortedUntilIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasIsSortedUntilIt<ForwardIteratorNotIncrementable>);
static_assert(!HasIsSortedUntilIt<int*, SentinelForNotSemiregular>);
static_assert(!HasIsSortedUntilIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasIsSortedUntilIt<HasNoComparator*>);

template <class Range>
concept HasIsSortedUntilR = requires(Range range) { std::ranges::is_sorted_until(range); };

static_assert(HasIsSortedUntilR<UncheckedRange<int*>>);
static_assert(!HasIsSortedUntilR<ForwardRangeNotDerivedFrom>);
static_assert(!HasIsSortedUntilR<ForwardRangeNotIncrementable>);
static_assert(!HasIsSortedUntilR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasIsSortedUntilR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasIsSortedUntilR<UncheckedRange<HasNoComparator*>>);

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 3};
      std::same_as<Iter> auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a + 5)));
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {1, 2, 3, 4, 3};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      std::same_as<Iter> auto ret = std::ranges::is_sorted_until(range);
      assert(base(ret) == a + 4);
    }
  }

  { // second element isn't sorted
    {
      int a[] = {1, 0, 3, 4, 5};
      auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a + 5)));
      assert(base(ret) == a + 1);
    }
    {
      int a[] = {1, 0, 3, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::is_sorted_until(range);
      assert(base(ret) == a + 1);
    }
  }

  { // all elements are sorted
    {
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a + 5)));
      assert(base(ret) == a + 5);
    }
    {
      int a[] = {1, 2, 3, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::is_sorted_until(range);
      assert(base(ret) == a + 5);
    }
  }

  { // check that the comparator is used
    {
      int a[] = {5, 4, 3, 2};
      auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a + 4)), std::ranges::greater{});
      assert(base(ret) == a + 4);
    }
    {
      int a[] = {5, 4, 3, 2};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::is_sorted_until(range, std::ranges::greater{});
      assert(base(ret) == a + 4);
    }
  }

  { // check that an empty range works
    {
      int a[] = {};
      auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a)));
      assert(base(ret) == a);
    }
    {
      int a[] = {};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a)));
      auto ret = std::ranges::is_sorted_until(range);
      assert(base(ret) == a);
    }
  }

  { // check that a range with a single element works
    {
      int a[] = {32};
      auto ret = std::ranges::is_sorted_until(Iter(a), Sent(Iter(a + 1)));
      assert(base(ret) == a + 1);
    }
    {
      int a[] = {32};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 1)));
      auto ret = std::ranges::is_sorted_until(range);
      assert(base(ret) == a + 1);
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
      auto ret = std::ranges::is_sorted_until(a, a + 3, {}, &S::i);
      assert(base(ret) == a + 3);
    }
    {
      S a[] = {1, 2, 3};
      auto ret = std::ranges::is_sorted_until(a, {}, &S::i);
      assert(base(ret) == a + 3);
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::is_sorted_until(std::array{1, 2, 3});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

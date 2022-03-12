//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<input_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr I ranges::find(I first, S last, const T& value, Proj proj = {});
// template<input_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr borrowed_iterator_t<R>
//     ranges::find(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class It, class Sent = It>
concept HasFindIt = requires(It it, Sent sent) { std::ranges::find(it, sent, *it); };
static_assert(HasFindIt<int*>);
static_assert(!HasFindIt<NotEqualityComparable*>);
static_assert(!HasFindIt<InputIteratorNotDerivedFrom>);
static_assert(!HasFindIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasFindIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasFindIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindIt<int*, int>);
static_assert(!HasFindIt<int, int*>);

template <class Range, class ValT>
concept HasFindR = requires(Range r) { std::ranges::find(r, ValT{}); };
static_assert(HasFindR<std::array<int, 1>, int>);
static_assert(!HasFindR<int, int>);
static_assert(!HasFindR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasFindR<InputRangeNotDerivedFrom, int>);
static_assert(!HasFindR<InputRangeNotIndirectlyReadable, int>);
static_assert(!HasFindR<InputRangeNotInputOrOutputIterator, int>);
static_assert(!HasFindR<InputRangeNotSentinelSemiregular, int>);
static_assert(!HasFindR<InputRangeNotSentinelEqualityComparableWith, int>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  {
    int a[] = {1, 2, 3, 4};
    std::same_as<It> auto ret = std::ranges::find(It(a), Sent(It(a + 4)), 4);
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
  {
    int a[] = {1, 2, 3, 4};
    auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
    std::same_as<It> auto ret = std::ranges::find(range, 4);
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
}

struct OneWayComparable {
  bool isLeft;
  friend constexpr bool operator==(OneWayComparable l, OneWayComparable) { return l.isLeft; }
};

struct NonConstComparableLValue {
  friend constexpr bool operator==(const NonConstComparableLValue&, const NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableLValue&, NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(const NonConstComparableLValue&, NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableLValue&, const NonConstComparableLValue&) { return true; }
};

struct NonConstComparableRValue {
  friend constexpr bool operator==(const NonConstComparableRValue&, const NonConstComparableRValue&) { return false; }
  friend constexpr bool operator==(const NonConstComparableRValue&&, const NonConstComparableRValue&&) { return false; }
  friend constexpr bool operator==(NonConstComparableRValue&&, NonConstComparableRValue&&) { return false; }
  friend constexpr bool operator==(NonConstComparableRValue&&, const NonConstComparableRValue&) { return true; }
};

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  {
    // check that projections are used properly and that they are called with the iterator directly
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::find(a, a + 4, a + 3, [](int& i) { return &i; });
      assert(ret == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::find(a, a + 3, [](int& i) { return &i; });
      assert(ret == a + 3);
    }
  }

  { // check that the first element is returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::find(a, 0, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::find(a, a + 3, 0, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
  }

  { // check that end + 1 iterator is returned with no match
    {
      int a[] = {1, 1, 1};
      auto ret = std::ranges::find(a, a + 3, 0);
      assert(ret == a + 3);
    }
    {
      int a[] = {1, 1, 1};
      auto ret = std::ranges::find(a, 0);
      assert(ret == a + 3);
    }
  }

  {
    // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
      std::ranges::find(std::array{1, 2}, 3);
  }

  {
    // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<int*> auto ret = std::ranges::find(std::views::all(a), 1);
    assert(ret == a);
    assert(*ret == 1);
  }

  {
    // check that std::invoke is used
    struct S { int i; };
    S a[] = { S{1}, S{3}, S{2} };
    std::same_as<S*> auto ret = std::ranges::find(a, 4, &S::i);
    assert(ret == a + 3);
  }

  {
    // count invocations of the projection
    {
      int a[] = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret = std::ranges::find(a, a + 4, 2, [&](int i) { ++projection_count; return i; });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 2);
    }
    {
      int a[] = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret = std::ranges::find(a, 2, [&](int i) { ++projection_count; return i; });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 2);
    }
  }

  {
    // check comparison order
    {
      OneWayComparable a[] = { OneWayComparable{true} };
      auto ret = std::ranges::find(a, a + 1, OneWayComparable{false});
      assert(ret == a);
    }
    {
      OneWayComparable a[] = { OneWayComparable{true} };
      auto ret = std::ranges::find(a, OneWayComparable{false});
      assert(ret == a);
    }
  }

  {
    // check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparableLValue a[] = { NonConstComparableLValue{} };
      auto ret = std::ranges::find(a, a + 1, NonConstComparableLValue{});
      assert(ret == a);
    }
    {
      using It = std::move_iterator<NonConstComparableRValue*>;
      NonConstComparableRValue a[] = { NonConstComparableRValue{} };
      auto ret = std::ranges::find(It(a), It(a + 1), NonConstComparableRValue{});
      assert(ret.base() == a);
    }
    {
      NonConstComparableLValue a[] = { NonConstComparableLValue{} };
      auto ret = std::ranges::find(a, NonConstComparableLValue{});
      assert(ret == a);
    }
    {
      using It = std::move_iterator<NonConstComparableRValue*>;
      NonConstComparableRValue a[] = { NonConstComparableRValue{} };
      auto range = std::ranges::subrange(It(a), It(a + 1));
      auto ret = std::ranges::find(range, NonConstComparableRValue{});
      assert(ret.base() == a);
    }
  }

  {
    // check that an empty range works
    {
      std::array<int ,0> a = {};
      auto ret = std::ranges::find(a.begin(), a.end(), 1);
      assert(ret == a.begin());
    }
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::find(a, 1);
      assert(ret == a.begin());
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//  constexpr subrange<I> ranges::find_last(I first, S last, const T& value, Proj proj = {});
// template<forward_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//  constexpr borrowed_subrange_t<R>
// ranges::find_last(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <concepts>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class It, class Sent = It>
concept HasFindIt = requires(It it, Sent sent) {
  std::ranges::find_last(it, sent, *it);
};
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
concept HasFindR = requires(Range r) {
  std::ranges::find_last(r, ValT{});
};

static_assert(HasFindR<std::array<int, 1>, int>);
static_assert(!HasFindR<int, int>);
static_assert(!HasFindR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasFindR<InputRangeNotDerivedFrom, int>);
static_assert(!HasFindR<InputRangeNotIndirectlyReadable, int>);
static_assert(!HasFindR<InputRangeNotInputOrOutputIterator, int>);
static_assert(!HasFindR<InputRangeNotSentinelSemiregular, int>);
static_assert(!HasFindR<InputRangeNotSentinelEqualityComparableWith, int>);

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

  {// check that projections are used properly and called with the reference to the element the iterator is pointing to
    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, a + 4, a + 3, [](int& i) { return &i; });
      assert(ret.data() == a + 3);
    }

    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, a + 3, [](int& i) { return &i; });
      assert(ret.data() == a + 3);
    }

  }

  {// check that the end element is returned

    {
      struct S{
        int comp;
        int other;
      };

      S a[] = {{0, 0}, {0, 2}, {0, 1}};
      std::same_as<std::ranges::borrowed_subrange_t<S (&)[3]>> auto ret = std::ranges::find_last(a, 0, &S::comp);
      assert(ret.data() == a + 2);
      assert(ret.data()->comp == 0);
      assert(ret.data()->other == 1);
    }

    {
      struct S {
        int comp;
        int other;
      };

      S a[] = {{0, 0}, {0, 2}, {0, 1}};
      std::same_as<std::ranges::subrange<S*>> auto ret = std::ranges::find_last(a, a + 3, 0, &S::comp);
      assert(ret.data() == a + 2);
      assert(ret.data()->comp == 0);
      assert(ret.data()->other == 1);
    }

  }

  {// check that end + 1 iterator is returned with no match

    {
      int a[] = {1, 1, 1};
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, a + 3, 0);
      assert(ret.data() == a + 3);
    }

    {
      int a[] = {1, 1, 1};
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, 0);
      assert(ret.data() == a + 3);
    }

  }

  {// check that ranges::dangling is returned
  [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret = std::ranges::find_last(std::array{1, 2}, 3);
  }

  {// check that an iterator is returned with a borrowing range
      int a[] = {1, 1, 2, 3, 4};
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(std::views::all(a), 1);
      assert(ret.data() == a + 1);
      assert(*(ret.data()) == 1);
  }

  {// check that std::invoke is used

    {
      struct S{
        int i;
      };

      S a[] = {S{1}, S{3}, S{2}};
      std::same_as<std::ranges::borrowed_subrange_t<S (&)[3]>> auto ret = std::ranges::find_last(a, 4, &S::i);
      assert(ret.data() == a + 3);
    }

    {
      struct S {
        int i;
      };

      S a[] = {S{1}, S{3}, S{2}};
      std::same_as<std::ranges::subrange<S*>> auto ret = std::ranges::find_last(a, a + 3, 4, &S::i);
      assert(ret.data() == a + 3);
    }
  }

  {// count invocations of the projection

     {
      int a[] = {1, 2, 2, 3, 4};
      int projection_count = 0;
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, a + 5, 2, [&](int i) { ++projection_count; return i; });
      assert(ret.data() == a + 2);
      assert(*(ret.data()) == 2);
      assert(projection_count == 3);
    }

    {
      int a[] = {1, 2, 2, 3, 4};
      int projection_count = 0;
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, 2, [&](int i) { ++projection_count; return i; });
      assert(ret.data() == a + 2);
      assert(*(ret.data()) == 2);
      assert(projection_count == 3);
    }

  }

  {// check comparison order

    {
      OneWayComparable a[] = {OneWayComparable{true}};
      std::same_as<std::ranges::subrange<OneWayComparable*>> auto ret = std::ranges::find_last(a, a + 1, OneWayComparable{false});
      assert(ret.data() == a);
    }

    {
      OneWayComparable a[] = {OneWayComparable{true}};
      std::same_as<std::ranges::borrowed_subrange_t<OneWayComparable(&)[1]>> auto ret = std::ranges::find_last(a, OneWayComparable{false});
      assert(ret.data() == a);
    }

  }

  {// check that the return type of `iter::operator*` doesn't change

    {
      NonConstComparableLValue a[] = {NonConstComparableLValue{}};
      std::same_as<std::ranges::subrange<NonConstComparableLValue*>> auto ret = std::ranges::find_last(a, a + 1, NonConstComparableLValue{});
      assert(ret.data() == a);
    }

    {
      using It = std::move_iterator<NonConstComparableRValue*>;
      NonConstComparableRValue a[] = {NonConstComparableRValue{}};
      std::same_as<std::ranges::subrange<std::move_iterator<NonConstComparableRValue*>>> auto ret = std::ranges::find_last(It(a), It(a + 1), NonConstComparableRValue{});
      assert(ret.begin().base() == a);
    }

    {
      NonConstComparableLValue a[] = {NonConstComparableLValue{}};
      std::same_as<std::ranges::borrowed_subrange_t<NonConstComparableLValue(&)[1]>> auto ret = std::ranges::find_last(a, NonConstComparableLValue{});
      assert(ret.data() == a);
    }

    {
      using It = std::move_iterator<NonConstComparableRValue*>;
      NonConstComparableRValue a[] = {NonConstComparableRValue{}};
      auto range = std::ranges::subrange(It(a), It(a + 1));
      std::same_as<std::ranges::borrowed_subrange_t<std::ranges::subrange<std::move_iterator<NonConstComparableRValue*>,
                                                                      std::move_iterator<NonConstComparableRValue*>,
                                                                      std::ranges::subrange_kind::sized>&>> auto ret = std::ranges::find_last(range, NonConstComparableRValue{});
      assert(ret.begin().base() == a);
    }

  }

  {// check that an empty range works
    {
      std::array<int, 0> a = {};
      int search_value = 1;
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a.begin(), a.end(), search_value);
      assert(ret.data() == a.end());
    }

    {
      std::array<int, 0> a = {};
      int search_value = 1;
      std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(a, search_value);
      assert(ret.data() == a.end());
    }

  }

  {// check that the implicit conversion to bool works

    {
      StrictComparable<int> a[] = {1, 2, 2, 3, 4};
      std::same_as<std::ranges::subrange<StrictComparable<int>*>> auto ret = std::ranges::find_last(a, a + 4, StrictComparable<int>{2});
      assert(ret.data() == a + 2);
    }

    {
      StrictComparable<int> a[] = {1, 2, 2, 3, 4};
      std::same_as<std::ranges::borrowed_subrange_t<StrictComparable<int>(&)[5]>> auto ret = std::ranges::find_last(a, StrictComparable<int>{2});
      assert(ret.data() == a + 2);
    }

  }

return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
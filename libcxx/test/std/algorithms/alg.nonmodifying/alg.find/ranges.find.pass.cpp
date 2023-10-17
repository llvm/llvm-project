//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS: -Wno-sign-compare

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
#include <vector>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
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

static std::vector<int> comparable_data;

template <class It, class Sent = It>
constexpr void test_iterators() {
  using ValueT = std::iter_value_t<It>;
  { // simple test
    {
      ValueT a[] = {1, 2, 3, 4};
      std::same_as<It> auto ret = std::ranges::find(It(a), Sent(It(a + 4)), 4);
      assert(base(ret) == a + 3);
      assert(*ret == 4);
    }
    {
      ValueT a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
      std::same_as<It> auto ret = std::ranges::find(range, 4);
      assert(base(ret) == a + 3);
      assert(*ret == 4);
    }
  }

  { // check that an empty range works
    {
      std::array<ValueT, 0> a = {};
      auto ret = std::ranges::find(It(a.data()), Sent(It(a.data())), 1);
      assert(base(ret) == a.data());
    }
    {
      std::array<ValueT, 0> a = {};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data())));
      auto ret = std::ranges::find(range, 1);
      assert(base(ret) == a.data());
    }
  }

  { // check that last is returned with no match
    {
      ValueT a[] = {1, 1, 1};
      auto ret = std::ranges::find(a, a + 3, 0);
      assert(ret == a + 3);
    }
    {
      ValueT a[] = {1, 1, 1};
      auto ret = std::ranges::find(a, 0);
      assert(ret == a + 3);
    }
  }

  if (!std::is_constant_evaluated())
    comparable_data.clear();
}

template <class ElementT>
class TriviallyComparable {
  ElementT el_;

public:
  TEST_CONSTEXPR TriviallyComparable(ElementT el) : el_(el) {}
  bool operator==(const TriviallyComparable&) const = default;
};

constexpr bool test() {
  types::for_each(types::type_list<char, wchar_t, int, long, TriviallyComparable<char>, TriviallyComparable<wchar_t>>{},
                  []<class T> {
                    types::for_each(types::cpp20_input_iterator_list<T*>{}, []<class Iter> {
                      if constexpr (std::forward_iterator<Iter>)
                        test_iterators<Iter>();
                      test_iterators<Iter, sentinel_wrapper<Iter>>();
                      test_iterators<Iter, sized_sentinel<Iter>>();
                    });
                  });

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

  {
    // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<int*> auto ret = std::ranges::find(std::views::all(a), 1);
    assert(ret == a);
    assert(*ret == 1);
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

  return true;
}

template <class IndexT>
class Comparable {
  IndexT index_;

public:
  Comparable(IndexT i)
      : index_([&]() {
          IndexT size = static_cast<IndexT>(comparable_data.size());
          comparable_data.push_back(i);
          return size;
        }()) {}

  bool operator==(const Comparable& other) const {
    return comparable_data[other.index_] == comparable_data[index_];
  }

  friend bool operator==(const Comparable& lhs, long long rhs) { return comparable_data[lhs.index_] == rhs; }
};

int main(int, char**) {
  test();
  static_assert(test());

  types::for_each(types::cpp20_input_iterator_list<Comparable<char>*>{}, []<class Iter> {
    if constexpr (std::forward_iterator<Iter>)
      test_iterators<Iter>();
    test_iterators<Iter, sentinel_wrapper<Iter>>();
    test_iterators<Iter, sized_sentinel<Iter>>();
  });

  types::for_each(types::cpp20_input_iterator_list<Comparable<wchar_t>*>{}, []<class Iter> {
    if constexpr (std::forward_iterator<Iter>)
      test_iterators<Iter>();
    test_iterators<Iter, sentinel_wrapper<Iter>>();
    test_iterators<Iter, sized_sentinel<Iter>>();
  });

  return 0;
}

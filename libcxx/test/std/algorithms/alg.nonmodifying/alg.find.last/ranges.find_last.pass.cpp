//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-sign-compare
// MSVC warning C4242: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// MSVC warning C4244: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4242 /wd4244

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr subrange<I> ranges::find_last(I first, S last, const T& value, Proj proj = {});
// template<forward_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr borrowed_subrange_t<R> ranges::find_last(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class It, class Sent = It>
concept HasFindLastIt = requires(It it, Sent sent) { std::ranges::find_last(it, sent, *it); };
static_assert(HasFindLastIt<int*>);
static_assert(HasFindLastIt<forward_iterator<int*>>);
static_assert(!HasFindLastIt<cpp20_input_iterator<int*>>);
static_assert(!HasFindLastIt<NotEqualityComparable*>);
static_assert(!HasFindLastIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIt<forward_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindLastIt<forward_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindLastIt<int*, int>);
static_assert(!HasFindLastIt<int, int*>);

template <class Range, class ValT>
concept HasFindLastR = requires(Range r) { std::ranges::find_last(r, ValT{}); };
static_assert(HasFindLastR<std::array<int, 1>, int>);
static_assert(!HasFindLastR<int, int>);
static_assert(!HasFindLastR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasFindLastR<ForwardRangeNotDerivedFrom, int>);
static_assert(!HasFindLastR<ForwardRangeNotIncrementable, int>);
static_assert(!HasFindLastR<ForwardRangeNotSentinelSemiregular, int>);
static_assert(!HasFindLastR<ForwardRangeNotSentinelEqualityComparableWith, int>);

static std::vector<int> comparable_data;

template <class It, class Sent = It>
constexpr void test_iterators() {
  using ValueT = std::iter_value_t<It>;
  { // simple test
    {
      ValueT a[] = {1, 2, 3, 4};

      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last(It(a), Sent(It(a + 4)), 2);
      assert(base(ret.begin()) == a + 1);
      assert(*ret.begin() == 2);
    }
    {
      ValueT a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));

      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last(range, 2);
      assert(base(ret.begin()) == a + 1);
      assert(*ret.begin() == 2);
    }
  }

  { // check that an empty range works
    {
      std::array<ValueT, 0> a = {};

      auto ret = std::ranges::find_last(It(a.data()), Sent(It(a.data())), 1).begin();
      assert(base(ret) == a.data());
    }
    {
      std::array<ValueT, 0> a = {};

      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data())));
      auto ret   = std::ranges::find_last(range, 1).begin();
      assert(base(ret) == a.data());
    }
  }

  { // check that last is returned with no match
    {
      ValueT a[] = {1, 1, 1};

      auto ret = std::ranges::find_last(a, a + 3, 0).begin();
      assert(ret == a + 3);
    }
    {
      ValueT a[] = {1, 1, 1};

      auto ret = std::ranges::find_last(a, 0).begin();
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
                    types::for_each(types::forward_iterator_list<T*>{}, []<class Iter> {
                      if constexpr (std::forward_iterator<Iter>)
                        test_iterators<Iter>();
                      test_iterators<Iter, sentinel_wrapper<Iter>>();
                      test_iterators<Iter, sized_sentinel<Iter>>();
                    });
                  });

  {
    std::vector<std::vector<int>> vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto view                         = vec | std::views::join;
    assert(std::ranges::find_last(view.begin(), view.end(), 4).begin() == std::next(view.begin(), 3));
    assert(std::ranges::find_last(view, 4).begin() == std::next(view.begin(), 3));
  }

  { // check that the last element is returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = std::ranges::find_last(a, 0, &S::comp).begin();
      assert(ret == a + 2);
      assert(ret->comp == 0);
      assert(ret->other == 1);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = std::ranges::find_last(a, a + 3, 0, &S::comp).begin();
      assert(ret == a + 2);
      assert(ret->comp == 0);
      assert(ret->other == 1);
    }
  }

  {
    // check that an iterator is returned with a borrowing range
    int a[]                                            = {1, 2, 3, 4};
    std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(std::views::all(a), 1);
    assert(ret.begin() == a);
    assert(*ret.begin() == 1);
  }

  {
    // count invocations of the projection
    {
      int a[]              = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret             = std::ranges::find_last(a, a + 4, 2, [&](int i) {
                   ++projection_count;
                   return i;
                 }).begin();
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 3);
    }
    {
      int a[]              = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret             = std::ranges::find_last(a, 2, [&](int i) {
                   ++projection_count;
                   return i;
                 }).begin();
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 3);
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

  bool operator==(const Comparable& other) const { return comparable_data[other.index_] == comparable_data[index_]; }

  friend bool operator==(const Comparable& lhs, long long rhs) { return comparable_data[lhs.index_] == rhs; }
};

void test_deque() {
  { // empty deque
    std::deque<int> data;
    assert(std::ranges::find_last(data, 4).begin() == data.end());
    assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.end());
  }

  { // single element - match
    std::deque<int> data = {4};
    assert(std::ranges::find_last(data, 4).begin() == data.begin());
    assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.begin());
  }

  { // single element - no match
    std::deque<int> data = {3};
    assert(std::ranges::find_last(data, 4).begin() == data.end());
    assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.end());
  }

  // many elements
  for (auto size : {2, 3, 1023, 1024, 1025, 2047, 2048, 2049}) {
    { // last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 1] = 4;
      assert(std::ranges::find_last(data, 4).begin() == data.end() - 1);
      assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.end() - 1);
    }

    { // second-last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 2] = 4;
      assert(std::ranges::find_last(data, 4).begin() == data.end() - 2);
      assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.end() - 2);
    }

    { // no match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      assert(std::ranges::find_last(data, 4).begin() == data.end());
      assert(std::ranges::find_last(data.begin(), data.end(), 4).begin() == data.end());
    }
  }
}

int main(int, char**) {
  test_deque();
  test();
  static_assert(test());

  types::for_each(types::forward_iterator_list<Comparable<char>*>{}, []<class Iter> {
    if constexpr (std::forward_iterator<Iter>)
      test_iterators<Iter>();
    test_iterators<Iter, sentinel_wrapper<Iter>>();
    test_iterators<Iter, sized_sentinel<Iter>>();
  });

  types::for_each(types::forward_iterator_list<Comparable<wchar_t>*>{}, []<class Iter> {
    if constexpr (std::forward_iterator<Iter>)
      test_iterators<Iter>();
    test_iterators<Iter, sentinel_wrapper<Iter>>();
    test_iterators<Iter, sized_sentinel<Iter>>();
  });

  return 0;
}

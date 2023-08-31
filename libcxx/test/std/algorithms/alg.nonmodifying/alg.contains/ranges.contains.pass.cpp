//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// template<input_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//     requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//     constexpr bool ranges::contains(I first, S last, const T& value, Proj proj = {});       // since C++23

// template<input_range R, class T, class Proj = identity>
//     requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//     constexpr bool ranges::contains(R&& r, const T& value, Proj proj = {});                 // since C++23

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class Iter, class Sent = Iter>
concept HasContainsIt = requires(Iter iter, Sent sent) { std::ranges::contains(iter, sent, *iter); };

static_assert(HasContainsIt<int*>);
static_assert(!HasContainsIt<NotEqualityComparable*>);
static_assert(!HasContainsIt<InputIteratorNotDerivedFrom>);
static_assert(!HasContainsIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasContainsIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasContainsIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasContainsIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasContainsIt<int*, int>);
static_assert(!HasContainsIt<int, int*>);

template <class Range, class ValT>
concept HasContainsR = requires(Range range) { std::ranges::contains(range, ValT{}); };

static_assert(HasContainsR<std::array<int, 1>, int>);
static_assert(!HasContainsR<int, int>);
static_assert(!HasContainsR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasContainsR<InputRangeNotDerivedFrom, int>);
static_assert(!HasContainsR<InputRangeNotIndirectlyReadable, int>);
static_assert(!HasContainsR<InputRangeNotInputOrOutputIterator, int>);
static_assert(!HasContainsR<InputRangeNotSentinelSemiregular, int>);
static_assert(!HasContainsR<InputRangeNotSentinelEqualityComparableWith, int>);

static std::vector<int> comparable_data;

// clang-format off
template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  using ValueT = std::iter_value_t<Iter>;
  {  // simple tests
    {
      ValueT a[] = {1, 2, 3, 4, 5, 6};
      std::same_as<bool> auto ret =
        std::ranges::contains(Iter(a), Sent(Iter(a + 6)), 3);
      assert(ret);
    }
    {
      ValueT a[] = {1, 2, 3, 4, 5, 6};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
      std::same_as<bool> decltype(auto) ret =
        std::ranges::contains(range, 3);
      assert(ret);
    }
  }

  { // check that an empty range works
    {
      ValueT a[] = {};
      auto ret = std::ranges::contains(Iter(a), Sent(Iter(a)), 1);
      assert(!ret);
    }
    {
      ValueT a[] = {};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a)));
      auto ret = std::ranges::contains(range, 1);
      assert(!ret);
    }
  }

  { // check that no match
    {
      ValueT a[] = {13, 1, 21, 4, 5};
      auto ret = std::ranges::contains(Iter(a), Sent(Iter(a + 5)), 10);
      assert(!ret);
    }
    {
      ValueT a[] = {13, 1, 21, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::contains(range, 10);
      assert(!ret);
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

constexpr bool test() {
  types::for_each(types::type_list<char, wchar_t, int, long, 
                  TriviallyComparable<char>, TriviallyComparable<wchar_t>>{},
                  []<class T> {
                    types::for_each(types::cpp20_input_iterator_list<T*>{}, 
                      []<class Iter> {
                      if constexpr (std::forward_iterator<Iter>)
                        test_iterators<Iter>();
                      test_iterators<Iter, sentinel_wrapper<Iter>>();
                      test_iterators<Iter, sized_sentinel<Iter>>();
                    });
                  });

  {
    // count invocations of the projection
    {
      int a[] = {1, 9, 0, 13, 25};
      int projection_count = 0;
      auto ret = std::ranges::contains(a, a + 5, 0,
                                [&](int i) { ++projection_count; return i; });
      assert(ret);
      assert(projection_count == 3);
    }
    {
      int a[] ={1, 9, 0, 13, 25};
      int projection_count = 0;
      auto range = std::ranges::subrange(a, a + 5);
      auto ret = std::ranges::contains(range, 0, [&](int i) { ++projection_count; return i; });
      assert(ret);
      assert(projection_count == 3);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  types::for_each(types::type_list<Comparable<char>, Comparable<wchar_t>>{},
                []<class T> {
                  types::for_each(types::cpp20_input_iterator_list<T*>{}, 
                    []<class Iter> {
                    if constexpr (std::forward_iterator<Iter>)
                      test_iterators<Iter>();
                    test_iterators<Iter, sentinel_wrapper<Iter>>();
                    test_iterators<Iter, sized_sentinel<Iter>>();
                  });
                });

  return 0;
}

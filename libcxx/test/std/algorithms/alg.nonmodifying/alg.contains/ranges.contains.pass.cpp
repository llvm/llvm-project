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
static_assert(!HasContainsIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);

static_assert(!HasContainsIt<int*, int>);
static_assert(!HasContainsIt<int, int*>);
static_assert(HasContainsIt<int*, int*>);

template <class Range, class ValT>
concept HasContainsR = requires(Range&& range) { std::ranges::contains(std::forward<Range>(range), ValT{}); };

static_assert(!HasContainsR<int, int>);
static_assert(HasContainsR<int[1], int>);
static_assert(!HasContainsR<NotEqualityComparable[1], NotEqualityComparable>);
static_assert(!HasContainsR<InputRangeNotDerivedFrom, int>);
static_assert(!HasContainsR<InputRangeNotIndirectlyReadable, int>);
static_assert(!HasContainsR<InputRangeNotInputOrOutputIterator, int>);
static_assert(!HasContainsR<InputRangeNotSentinelSemiregular, int>);
static_assert(!HasContainsR<InputRangeNotSentinelEqualityComparableWith, int>);

static std::vector<int> comparable_data;

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  using ValueT = std::iter_value_t<Iter>;
  {  // simple tests
    ValueT a[] = {1, 2, 3, 4, 5, 6};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret =
        std::ranges::contains(whole.begin(), whole.end(), 3);
      assert(ret);
    }
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret =
        std::ranges::contains(whole, 3);
      assert(ret);
    }
  }

  { // check that a range with a single element works
    ValueT a[] = {32};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 1)));
    {
      bool ret = std::ranges::contains(whole.begin(), whole.end(), 32);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains(whole, 32);
      assert(ret);
    }
  }

  { // check that an empty range works
    ValueT a[] = {};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a)));
    {
      bool ret = std::ranges::contains(whole.begin(), whole.end(), 1);
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains(whole, 1);
      assert(!ret);
    }
  }

  { // check that the first element matches
    ValueT a[] = {32, 3, 2, 1, 0, 23, 21, 9, 40, 100};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 10)));
    {
      bool ret = std::ranges::contains(whole.begin(), whole.end(), 32);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains(whole, 32);
      assert(ret);
    }
  }

  { // check that the last element matches
    ValueT a[] = {3, 22, 1, 43, 99, 0, 56, 100, 32};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 9)));
    {
      bool ret = std::ranges::contains(whole.begin(), whole.end(), 32);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains(whole, 32);
      assert(ret);
    }
  }

  { // no match
    ValueT a[] = {13, 1, 21, 4, 5};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
    {
      bool ret = std::ranges::contains(whole.begin(), whole.end(), 10);
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains(whole, 10);
      assert(!ret);
    }
  }

  { // check that the projections are used
    int a[] = {1, 9, 0, 13, 25};
    {
      bool ret = std::ranges::contains(a, a + 5, -13, [&](int i) { return i * -1; });
      assert(ret);
    }
    {
      auto range = std::ranges::subrange(a, a + 5);
      bool ret = std::ranges::contains(range, -13, [&](int i) { return i * -1; });
      assert(ret);
    }
  }

  { // check the nodiscard extension
    // use #pragma around to suppress error: ignoring return value of function
    // declared with 'nodiscard' attribute [-Werror,-Wunused-result]
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-result"
    ValueT a[] = {1, 9, 0, 13, 25};
    auto whole = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
    std::ranges::contains(whole, 12);
    #pragma clang diagnostic pop
  }

  if (!std::is_constant_evaluated())
    comparable_data.clear();
}

template <class ElementT>
class TriviallyComparable {
  ElementT el_;

public:
  constexpr TriviallyComparable(ElementT el) : el_(el) {}
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
  types::for_each(types::type_list<char, short, int, long, long long,
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

  { // count invocations of the projection
    int a[] = {1, 9, 0, 13, 25};
    int projection_count = 0;
    {
      bool ret = std::ranges::contains(a, a + 5, 0,
                                [&](int i) { ++projection_count; return i; });
      assert(ret);
      assert(projection_count == 3);
    }
    {
      projection_count = 0;
      auto range = std::ranges::subrange(a, a + 5);
      bool ret = std::ranges::contains(range, 0, [&](int i) { ++projection_count; return i; });
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-sign-compare
// MSVC warning C4242: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// MSVC warning C4244: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4242 /wd4244
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=20000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=80000000

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
#include <cstddef>
#include <deque>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "sized_allocator.h"
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
      ValueT a[]                = {1, 2, 3, 4};
      std::same_as<It> auto ret = std::ranges::find(It(a), Sent(It(a + 4)), 4);
      assert(base(ret) == a + 3);
      assert(*ret == 4);
    }
    {
      ValueT a[]                = {1, 2, 3, 4};
      auto range                = std::ranges::subrange(It(a), Sent(It(a + 4)));
      std::same_as<It> auto ret = std::ranges::find(range, 4);
      assert(base(ret) == a + 3);
      assert(*ret == 4);
    }
  }

  { // check that an empty range works
    {
      std::array<ValueT, 0> a = {};
      auto ret                = std::ranges::find(It(a.data()), Sent(It(a.data())), 1);
      assert(base(ret) == a.data());
    }
    {
      std::array<ValueT, 0> a = {};
      auto range              = std::ranges::subrange(It(a.data()), Sent(It(a.data())));
      auto ret                = std::ranges::find(range, 1);
      assert(base(ret) == a.data());
    }
  }

  { // check that last is returned with no match
    {
      ValueT a[] = {1, 1, 1};
      auto ret   = std::ranges::find(a, a + 3, 0);
      assert(ret == a + 3);
    }
    {
      ValueT a[] = {1, 1, 1};
      auto ret   = std::ranges::find(a, 0);
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

#if TEST_STD_VER >= 20
  {
    std::vector<std::vector<int>> vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto view                         = vec | std::views::join;
    assert(std::ranges::find(view.begin(), view.end(), 4) == std::next(view.begin(), 3));
    assert(std::ranges::find(view, 4) == std::next(view.begin(), 3));
  }
#endif

  { // check that the first element is returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
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
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = std::ranges::find(a, a + 3, 0, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
  }

  {
    // check that an iterator is returned with a borrowing range
    int a[]                     = {1, 2, 3, 4};
    std::same_as<int*> auto ret = std::ranges::find(std::views::all(a), 1);
    assert(ret == a);
    assert(*ret == 1);
  }

  {
    // count invocations of the projection
    {
      int a[]              = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret             = std::ranges::find(a, a + 4, 2, [&](int i) {
        ++projection_count;
        return i;
      });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 2);
    }
    {
      int a[]              = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret             = std::ranges::find(a, 2, [&](int i) {
        ++projection_count;
        return i;
      });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(projection_count == 2);
    }
  }

  {
    // Test vector<bool>::iterator optimization
    std::vector<bool> vec(256 + 8);
    for (ptrdiff_t i = 8; i <= 256; i *= 2) {
      for (size_t offset = 0; offset < 8; offset += 2) {
        std::fill(vec.begin(), vec.end(), false);
        std::fill(vec.begin() + offset, vec.begin() + i + offset, true);

        // check both range and iterator-sentinel overloads
        assert(std::ranges::find(vec, true) == std::ranges::begin(vec) + offset);
        assert(std::ranges::find(std::ranges::begin(vec) + offset, std::ranges::end(vec), false) ==
               std::ranges::begin(vec) + offset + i);
      }
    }

    // Verify that the std::vector<bool>::iterator optimization works properly for allocators with custom size types
    // See https://llvm.org/PR122528
    {
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(100, false, Alloc(1));
      in[in.size() - 2] = true;
      assert(std::ranges::find(in, true) == in.end() - 2);
    }
    {
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(199, false, Alloc(1));
      in[in.size() - 2] = true;
      assert(std::ranges::find(in, true) == in.end() - 2);
    }
    {
      using Alloc = sized_allocator<bool, unsigned short, short>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      in[in.size() - 2] = true;
      assert(std::ranges::find(in, true) == in.end() - 2);
    }
    {
      using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
      std::vector<bool, Alloc> in(205, false, Alloc(1));
      in[in.size() - 2] = true;
      assert(std::ranges::find(in, true) == in.end() - 2);
    }
    {
      using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
      std::vector<bool, Alloc> in(257, false, Alloc(1));
      in[in.size() - 2] = true;
      assert(std::ranges::find(in, true) == in.end() - 2);
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
    assert(std::ranges::find(data, 4) == data.end());
    assert(std::ranges::find(data.begin(), data.end(), 4) == data.end());
  }

  { // single element - match
    std::deque<int> data = {4};
    assert(std::ranges::find(data, 4) == data.begin());
    assert(std::ranges::find(data.begin(), data.end(), 4) == data.begin());
  }

  { // single element - no match
    std::deque<int> data = {3};
    assert(std::ranges::find(data, 4) == data.end());
    assert(std::ranges::find(data.begin(), data.end(), 4) == data.end());
  }

  // many elements
  for (auto size : {2, 3, 1023, 1024, 1025, 2047, 2048, 2049}) {
    { // last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 1] = 4;
      assert(std::ranges::find(data, 4) == data.end() - 1);
      assert(std::ranges::find(data.begin(), data.end(), 4) == data.end() - 1);
    }

    { // second-last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 2] = 4;
      assert(std::ranges::find(data, 4) == data.end() - 2);
      assert(std::ranges::find(data.begin(), data.end(), 4) == data.end() - 2);
    }

    { // no match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      assert(std::ranges::find(data, 4) == data.end());
      assert(std::ranges::find(data.begin(), data.end(), 4) == data.end());
    }
  }
}

int main(int, char**) {
  test_deque();
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

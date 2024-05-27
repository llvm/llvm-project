//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=20000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=80000000

// template<input_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr iter_difference_t<I>
//     ranges::count(I first, S last, const T& value, Proj proj = {});
// template<input_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr range_difference_t<R>
//     ranges::count(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotEqualityComparable {
  bool operator==(NotEqualityComparable&&) const;
  bool operator==(NotEqualityComparable&) const;
  bool operator==(const NotEqualityComparable&&) const;
};

template <class It, class Sent = It>
concept HasCountIt = requires(It it, Sent sent) { std::ranges::count(it, sent, *it); };
static_assert(HasCountIt<int*>);
static_assert(!HasCountIt<NotEqualityComparable*>);
static_assert(!HasCountIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCountIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCountIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCountIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasCountIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasCountIt<int*, int>);
static_assert(!HasCountIt<int, int*>);

template <class Range, class ValT>
concept HasCountR = requires(Range r) { std::ranges::count(r, ValT{}); };
static_assert(HasCountR<std::array<int, 1>, int>);
static_assert(!HasCountR<int, int>);
static_assert(!HasCountR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasCountR<InputRangeNotDerivedFrom, int>);
static_assert(!HasCountR<InputRangeNotIndirectlyReadable, int>);
static_assert(!HasCountR<InputRangeNotInputOrOutputIterator, int>);
static_assert(!HasCountR<InputRangeNotSentinelSemiregular, int>);
static_assert(!HasCountR<InputRangeNotSentinelEqualityComparableWith, int>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  {
    // simple test
    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ptrdiff_t> auto ret = std::ranges::count(It(a), Sent(It(a + 4)), 3);
      assert(ret == 1);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
      std::same_as<std::ptrdiff_t> auto ret = std::ranges::count(range, 3);
      assert(ret == 1);
    }
  }

  {
    // check that an empty range works
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::count(It(a.data()), Sent(It(a.data() + a.size())), 1);
      assert(ret == 0);
    }
    {
      std::array<int, 0> a = {};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count(range, 1);
      assert(ret == 0);
    }
  }

  {
    // check that a range with a single element works
    {
      std::array a = {2};
      auto ret = std::ranges::count(It(a.data()), Sent(It(a.data() + a.size())), 2);
      assert(ret == 1);
    }
    {
      std::array a = {2};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count(range, 2);
      assert(ret == 1);
    }
  }

  {
    // check that 0 is returned with no match
    {
      std::array a = {1, 1, 1};
      auto ret = std::ranges::count(It(a.data()), Sent(It(a.data() + a.size())), 0);
      assert(ret == 0);
    }
    {
      std::array a = {1, 1, 1};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count(range, 0);
      assert(ret == 0);
    }
  }

  {
    // check that more than one element is counted
    {
      std::array a = {3, 3, 4, 3, 3};
      auto ret = std::ranges::count(It(a.data()), Sent(It(a.data() + a.size())), 3);
      assert(ret == 4);
    }
    {
      std::array a = {3, 3, 4, 3, 3};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count(range, 3);
      assert(ret == 4);
    }
  }

  {
    // check that all elements are counted
    {
      std::array a = {5, 5, 5, 5};
      auto ret = std::ranges::count(It(a.data()), Sent(It(a.data() + a.size())), 5);
      assert(ret == 4);
    }
    {
      std::array a = {5, 5, 5, 5};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count(range, 5);
      assert(ret == 4);
    }
  }
}

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
      auto ret = std::ranges::count(a, a + 4, a + 3, [](int& i) { return &i; });
      assert(ret == 1);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::count(a, a + 3, [](int& i) { return &i; });
      assert(ret == 1);
    }
  }

  {
    // check that std::invoke is used
    struct S { int i; };
    S a[] = { S{1}, S{3}, S{2} };
    std::same_as<std::ptrdiff_t> auto ret = std::ranges::count(a, 4, &S::i);
    assert(ret == 0);
  }

  {
    // count invocations of the projection
    {
      int a[] = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret = std::ranges::count(a, a + 4, 2, [&](int i) { ++projection_count; return i; });
      assert(ret == 1);
      assert(projection_count == 4);
    }
    {
      int a[] = {1, 2, 3, 4};
      int projection_count = 0;
      auto ret = std::ranges::count(a, 2, [&](int i) { ++projection_count; return i; });
      assert(ret == 1);
      assert(projection_count == 4);
    }
  }

  {
    // check that an immobile type works
    struct NonMovable {
      NonMovable(const NonMovable&) = delete;
      NonMovable(NonMovable&&) = delete;
      constexpr NonMovable(int i_) : i(i_) {}
      int i;

      bool operator==(const NonMovable&) const = default;
    };
    {
      NonMovable a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count(a, a + 4, NonMovable(8));
      assert(ret == 1);
    }
    {
      NonMovable a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count(a, NonMovable(8));
      assert(ret == 1);
    }
  }

  {
    // check that difference_type is used
    struct DiffTypeIterator {
      using difference_type = signed char;
      using value_type = int;

      int* it = nullptr;

      constexpr DiffTypeIterator() = default;
      constexpr DiffTypeIterator(int* i) : it(i) {}

      constexpr int& operator*() const { return *it; }
      constexpr DiffTypeIterator& operator++() { ++it; return *this; }
      constexpr void operator++(int) { ++it; }

      bool operator==(const DiffTypeIterator&) const = default;
    };

    {
      int a[] = {5, 5, 4, 3, 2, 1};
      std::same_as<signed char> decltype(auto) ret =
          std::ranges::count(DiffTypeIterator(a), DiffTypeIterator(a + 6), 4);
      assert(ret == 1);
    }
    {
      int a[] = {5, 5, 4, 3, 2, 1};
      auto range = std::ranges::subrange(DiffTypeIterator(a), DiffTypeIterator(a + 6));
      std::same_as<signed char> decltype(auto) ret = std::ranges::count(range, 4);
      assert(ret == 1);
    }
  }

  { // check that __bit_iterator optimizations work as expected
    std::vector<bool> vec(256 + 64);
    for (ptrdiff_t i = 0; i != 256; ++i) {
      for (size_t offset = 0; offset != 64; ++offset) {
        std::fill(vec.begin(), vec.end(), false);
        std::fill(vec.begin() + offset, vec.begin() + i + offset, true);
        assert(std::ranges::count(vec.begin() + offset, vec.begin() + offset + 256, true) == i);
        assert(std::ranges::count(vec.begin() + offset, vec.begin() + offset + 256, false) == 256 - i);
      }
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

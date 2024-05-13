//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

//  template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//    indirect_strict_weak_order<projected<I, Proj>> Comp = ranges::less>
//  constexpr ranges::minmax_element_result<I> ranges::minmax_element(I first, S last, Comp comp = {}, Proj proj = {});
//
//  template<forward_range R, class Proj = identity,
//    indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//  constexpr ranges::minmax_element_result<borrowed_iterator_t<R>>
//    ranges::minmax_element(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class T>
concept HasMinMaxElementR = requires(T t) { std::ranges::minmax_element(t); };

struct NoLessThanOp {};
struct NotTotallyOrdered {
  int i;
  bool operator<(const NotTotallyOrdered& o) const { return i < o.i; }
};

static_assert(HasMinMaxElementR<int (&)[10]>); // make sure HasMinMaxElementR works with an array
static_assert(!HasMinMaxElementR<NoLessThanOp (&)[10]>);
static_assert(!HasMinMaxElementR<NotTotallyOrdered (&)[10]>);

static_assert(HasMinMaxElementR<std::initializer_list<int>>); // make sure HasMinMaxElementR works with an initializer_list
static_assert(!HasMinMaxElementR<std::initializer_list<NoLessThanOp>>);
static_assert(!HasMinMaxElementR<std::initializer_list<NotTotallyOrdered>>);
static_assert(!HasMinMaxElementR<InputRangeNotDerivedFrom>);
static_assert(!HasMinMaxElementR<InputRangeNotIndirectlyReadable>);
static_assert(!HasMinMaxElementR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasMinMaxElementR<InputRangeNotSentinelSemiregular>);
static_assert(!HasMinMaxElementR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = sentinel_wrapper<It>>
concept HasMinMaxElementIt = requires(It it, Sent sent) { std::ranges::minmax_element(it, sent); };
static_assert(HasMinMaxElementIt<int*>); // make sure HasMinMaxElementIt works
static_assert(!HasMinMaxElementIt<InputIteratorNotDerivedFrom>);
static_assert(!HasMinMaxElementIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasMinMaxElementIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasMinMaxElementIt<int*, SentinelForNotSemiregular>);
static_assert(!HasMinMaxElementIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

static_assert(std::is_same_v<std::ranges::minmax_element_result<int>, std::ranges::min_max_result<int>>);

template <class It>
constexpr void test_iterators(std::initializer_list<int> a, int expectedMin, int expectedMax) {
  using Expected = std::ranges::minmax_element_result<It>;
  const int* first = a.begin();
  const int* last = a.end();
  {
    std::same_as<Expected> auto it = std::ranges::minmax_element(It(first), It(last));
    assert(base(it.min) == first + expectedMin);
    assert(base(it.max) == first + expectedMax);
  }
  {
    using Sent = sentinel_wrapper<It>;
    std::same_as<Expected> auto it = std::ranges::minmax_element(It(first), Sent(It(last)));
    assert(base(it.min) == first + expectedMin);
    assert(base(it.max) == first + expectedMax);
  }
  {
    auto range = std::ranges::subrange(It(first), It(last));
    std::same_as<Expected> auto it = std::ranges::minmax_element(range);
    assert(base(it.min) == first + expectedMin);
    assert(base(it.max) == first + expectedMax);
  }
  {
    using Sent = sentinel_wrapper<It>;
    auto range = std::ranges::subrange(It(first), Sent(It(last)));
    std::same_as<Expected> auto it = std::ranges::minmax_element(range);
    assert(base(it.min) == first + expectedMin);
    assert(base(it.max) == first + expectedMax);
  }
}

template <class It>
constexpr bool test_iterators() {
  test_iterators<It>({}, 0, 0);
  test_iterators<It>({1}, 0, 0);
  test_iterators<It>({1, 2}, 0, 1);
  test_iterators<It>({2, 1}, 1, 0);
  test_iterators<It>({2, 1, 2}, 1, 2);
  test_iterators<It>({2, 1, 1}, 1, 0);
  test_iterators<It>({2, 2, 1}, 2, 1);

  return true;
}

constexpr void test_borrowed_range_and_sentinel() {
  int a[] = {7, 6, 1, 3, 5, 1, 2, 4};

  std::ranges::minmax_element_result<int*> ret = std::ranges::minmax_element(std::views::all(a));
  assert(ret.min == a + 2);
  assert(ret.max == a + 0);
  assert(*ret.min == 1);
  assert(*ret.max == 7);
}

constexpr void test_comparator() {
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  std::ranges::minmax_element_result<int*> ret = std::ranges::minmax_element(a, std::ranges::greater{});
  assert(ret.min == a + 2);
  assert(ret.max == a + 5);
  assert(*ret.min == 9);
  assert(*ret.max == 1);
}

constexpr void test_projection() {
  {
    int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
    std::ranges::minmax_element_result<int*> ret =
        std::ranges::minmax_element(a, std::ranges::less{}, [](int i) { return i == 5 ? -100 : i; });
    assert(ret.min == a + 4);
    assert(ret.max == a + 2);
    assert(*ret.min == 5);
    assert(*ret.max == 9);
  }
  {
    int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
    std::ranges::minmax_element_result<int*> ret =
        std::ranges::minmax_element(a, std::less<int*>{}, [](int& i) { return &i; });
    assert(ret.min == a + 0);
    assert(ret.max == a + 7);
    assert(*ret.min == 7);
    assert(*ret.max == 4);
  }
}

struct Immobile {
  int i;

  constexpr Immobile(int i_) : i(i_) {}
  Immobile(const Immobile&) = delete;
  Immobile(Immobile&&) = delete;

  auto operator<=>(const Immobile&) const = default;
};

constexpr void test_immobile() {
  {
    Immobile arr[]{1, 2, 3};
    auto ret = std::ranges::minmax_element(arr);
    assert(ret.min == arr + 0);
    assert(ret.max == arr + 2);
  }
  {
    Immobile arr[]{1, 2, 3};
    auto ret = std::ranges::minmax_element(arr, arr + 3);
    assert(ret.min == arr + 0);
    assert(ret.max == arr + 2);
  }
}

constexpr void test_dangling() {
  int compares = 0;
  int projections = 0;
  auto comparator = [&](int a, int b) {
    ++compares;
    return a < b;
  };
  auto projection = [&](int a) {
    ++projections;
    return a;
  };
  [[maybe_unused]] std::same_as<std::ranges::minmax_element_result<std::ranges::dangling>> auto ret =
      std::ranges::minmax_element(std::array{1, 2, 3}, comparator, projection);
  assert(compares == 3);
  assert(projections == 6);
}

constexpr bool test() {
  test_iterators<forward_iterator<const int*>>();
  test_iterators<bidirectional_iterator<const int*>>();
  test_iterators<random_access_iterator<const int*>>();
  test_iterators<contiguous_iterator<const int*>>();
  test_iterators<const int*>();
  test_iterators<int*>();

  test_borrowed_range_and_sentinel();
  test_comparator();
  test_projection();
  test_dangling();

  { // check that std::invoke is used
    {
      struct S {
        int i;
      };
      S b[3] = {S{2}, S{1}, S{3}};
      std::same_as<std::ranges::minmax_element_result<S*>> auto ret = std::ranges::minmax_element(b, {}, &S::i);
      assert(ret.min->i == 1);
      assert(ret.max->i == 3);
      assert(ret.min == b + 1);
      assert(ret.max == b + 2);
    }
    {
      struct S {
        int i;
      };
      S b[3] = {S{2}, S{1}, S{3}};
      std::same_as<std::ranges::minmax_element_result<S*>> auto ret = std::ranges::minmax_element(b, b + 3, {}, &S::i);
      assert(ret.min->i == 1);
      assert(ret.max->i == 3);
      assert(ret.min == b + 1);
      assert(ret.max == b + 2);
    }
  }

  { // check that the leftmost value for min an rightmost for max are returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::minmax_element(a, a + 3, {}, &S::comp);
      assert(ret.min->comp == 0);
      assert(ret.max->comp == 0);
      assert(ret.min->other == 0);
      assert(ret.max->other == 1);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::minmax_element(a, {}, &S::comp);
      assert(ret.min->comp == 0);
      assert(ret.max->comp == 0);
      assert(ret.min->other == 0);
      assert(ret.max->other == 1);
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::minmax_element(a.begin(), a.end());
      assert(ret.min == a.begin());
      assert(ret.max == a.begin());
    }
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::minmax_element(a);
      assert(ret.min == a.begin());
      assert(ret.max == a.begin());
    }
  }

  { // check in ascending order
    {
      int a[] = {1, 2, 3};
      auto ret = std::ranges::minmax_element(a, a + 3);
      assert(ret.min == a + 0);
      assert(ret.max == a + 2);
      assert(*ret.min == 1);
      assert(*ret.max == 3);
    }
    {
      int a[] = {1, 2, 3};
      auto ret = std::ranges::minmax_element(a);
      assert(ret.min == a + 0);
      assert(ret.max == a + 2);
      assert(*ret.min == 1);
      assert(*ret.max == 3);
    }
  }

  { // check in descending order
    {
      int a[] = {3, 2, 1};
      auto ret = std::ranges::minmax_element(a, a + 3);
      assert(ret.min == a + 2);
      assert(ret.max == a + 0);
      assert(*ret.min == 1);
      assert(*ret.max == 3);
    }
    {
      int a[] = {3, 2, 1};
      auto ret = std::ranges::minmax_element(a);
      assert(ret.min == a + 2);
      assert(ret.max == a + 0);
      assert(*ret.min == 1);
      assert(*ret.max == 3);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

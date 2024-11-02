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

//  template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//    indirect_strict_weak_order<projected<I, Proj>> Comp = ranges::less>
//  constexpr I ranges::max_element(I first, S last, Comp comp = {}, Proj proj = {});
//
//  template<forward_range R, class Proj = identity,
//    indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//  constexpr borrowed_iterator_t<R> ranges::max_element(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <random>
#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"

template <class T>
concept HasMaxElement = requires (T t) { std::ranges::max_element(t); };

struct NoLessThanOp {};
struct NotTotallyOrdered {
  int i;
  bool operator<(const NotTotallyOrdered& o) const { return i < o.i; }
};

static_assert(HasMaxElement<std::array<int, 0>>);
static_assert(!HasMaxElement<int>);
static_assert(!HasMaxElement<NoLessThanOp>);
static_assert(!HasMaxElement<NotTotallyOrdered>);

template <class Iter>
constexpr void test_iterators(Iter first, Iter last) {
  std::same_as<Iter> auto it = std::ranges::max_element(first, last);
  if (first != last) {
    for (Iter j = first; j != last; ++j)
      assert(!(*j > *it));
  } else {
    assert(it == first);
  }
}

template <class Range, class Iter>
constexpr void test_range(Range&& rng, Iter begin, Iter end) {
  std::same_as<Iter> auto it = std::ranges::max_element(std::forward<Range>(rng));
  if (begin != end) {
    for (Iter j = begin; j != end; ++j)
      assert(!(*j > *it));
  } else {
    assert(it == begin);
  }
}

template <class It>
constexpr void test(std::initializer_list<int> a, int expected) {
  const int* first = a.begin();
  const int* last = a.end();
  {
    std::same_as<It> auto it = std::ranges::max_element(It(first), It(last));
    assert(base(it) == first + expected);
  }
  {
    using Sent = sentinel_wrapper<It>;
    std::same_as<It> auto it = std::ranges::max_element(It(first), Sent(It(last)));
    assert(base(it) == first + expected);
  }
  {
    auto range = std::ranges::subrange(It(first), It(last));
    std::same_as<It> auto it = std::ranges::max_element(range);
    assert(base(it) == first + expected);
  }
  {
    using Sent = sentinel_wrapper<It>;
    auto range = std::ranges::subrange(It(first), Sent(It(last)));
    std::same_as<It> auto it = std::ranges::max_element(range);
    assert(base(it) == first + expected);
  }
}

template <class It>
constexpr bool test() {
  test<It>({}, 0);
  test<It>({1}, 0);
  test<It>({1, 2}, 1);
  test<It>({2, 1}, 0);
  test<It>({2, 1, 2}, 0);
  test<It>({2, 1, 1}, 0);

  return true;
}

constexpr void test_borrowed_range_and_sentinel() {
  int a[] = {7, 6, 1, 3, 5, 1, 2, 4};

  int* ret = std::ranges::max_element(std::views::all(a));
  assert(ret == a + 0);
  assert(*ret == 7);
}

constexpr void test_comparator() {
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  int* ret = std::ranges::max_element(a, std::ranges::greater{});
  assert(ret == a + 5);
  assert(*ret == 1);
}

constexpr void test_projection() {
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  {
    int* ret = std::ranges::max_element(a, std::ranges::less{}, [](int i) { return i == 5 ? 100 : i; });
    assert(ret == a + 4);
    assert(*ret == 5);
  }
  {
    int* ret = std::ranges::max_element(a, std::less<int*>{}, [](int& i) { return &i; });
    assert(ret == a + 7);
    assert(*ret == 4);
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

  Immobile arr[] {1, 2, 3};
  assert(std::ranges::max_element(arr) == arr);
  assert(std::ranges::max_element(arr, arr + 3) == arr);
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
  [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
      std::ranges::max_element(std::array{1, 2, 3}, comparator, projection);
  assert(compares == 2);
  assert(projections == 4);
}

constexpr bool test() {

  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  int a[] = {7, 6, 5, 3, 4, 2, 1, 8};
  test_iterators(a, a + 8);
  int a2[] = {7, 6, 5, 3, 4, 2, 1, 8};
  test_range(a2, a2, a2 + 8);

  test_borrowed_range_and_sentinel();
  test_comparator();
  test_projection();
  test_dangling();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

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

// template<class T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr const T& ranges::max(const T& a, const T& b, Comp comp = {}, Proj proj = {});
//
// template<copyable T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr T ranges::max(initializer_list<T> r, Comp comp = {}, Proj proj = {});
//
// template<input_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   requires indirectly_copyable_storable<iterator_t<R>, range_value_t<R>*>
//   constexpr range_value_t<R>
//     ranges::max(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T>
concept HasMaxR = requires { std::ranges::max(std::declval<T>()); };

struct NoLessThanOp {};
struct NotTotallyOrdered {
  int i;
  bool operator<(const NotTotallyOrdered& o) const { return i < o.i; }
};

struct Movable {
  Movable& operator=(Movable&&) = default;
  Movable(Movable&&) = default;
  Movable(const Movable&) = delete;
};

static_assert(!HasMaxR<int>);

static_assert(HasMaxR<int(&)[10]>);
static_assert(HasMaxR<int(&&)[10]>);
static_assert(!HasMaxR<NoLessThanOp(&)[10]>);
static_assert(!HasMaxR<NotTotallyOrdered(&)[10]>);
static_assert(!HasMaxR<Movable(&)[10]>);

static_assert(HasMaxR<std::initializer_list<int>>);
static_assert(!HasMaxR<std::initializer_list<NoLessThanOp>>);
static_assert(!HasMaxR<std::initializer_list<NotTotallyOrdered>>);
static_assert(!HasMaxR<std::initializer_list<Movable>>);
static_assert(!HasMaxR<InputRangeNotDerivedFrom>);
static_assert(!HasMaxR<InputRangeNotIndirectlyReadable>);
static_assert(!HasMaxR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasMaxR<InputRangeNotSentinelSemiregular>);
static_assert(!HasMaxR<InputRangeNotSentinelEqualityComparableWith>);

template <class T, class U = T>
concept HasMax2 = requires { std::ranges::max(std::declval<T>(), std::declval<U>()); };

static_assert(HasMax2<int>);
static_assert(!HasMax2<int, long>);

static_assert(std::is_same_v<decltype(std::ranges::max(1, 2)), const int&>);

constexpr void test_2_arguments() {
  assert(std::ranges::max(1, 2) == 2);
  assert(std::ranges::max(2, 1) == 2);
  // test comparator
  assert(std::ranges::max(1, 2, std::ranges::greater{}) == 1);
  // test projection
  assert(std::ranges::max(1, 2, std::ranges::less{}, [](int i){ return i == 1 ? 10 : i; }) == 1);

  { // check that std::invoke is used
    struct S { int i; };
    S a[3] = { S{2}, S{1}, S{3} };
    decltype(auto) ret = std::ranges::max(a[0], a[1], {}, &S::i);
    ASSERT_SAME_TYPE(decltype(ret), const S&);
    assert(&ret == &a[0]);
    assert(ret.i == 2);
  }

  { // check that pointers are compared and not a range
    int i[1];
    int* a[] = {i, i + 1};
    auto ret = std::ranges::max(a[0], a[1]);
    assert(ret == i + 1);
  }

  { // test predicate and projection count
    int compares = 0;
    int projections = 0;
    auto comparator = [&](int x, int y) {
      ++compares;
      return x < y;
    };
    auto projection = [&](int x) {
      ++projections;
      return x;
    };
    auto ret = std::ranges::max(1, 2, comparator, projection);
    assert(ret == 2);
    assert(compares == 1);
    assert(projections == 2);
  }

  { // check that the first argument is returned
    struct S { int check; int other; };
    auto ret = std::ranges::max(S {0, 1}, S {0, 2}, {}, &S::check);
    assert(ret.other == 1);
  }
}

constexpr void test_initializer_list() {
  { // test projection
    auto proj = [](int i) { return i == 5 ? 100 : i; };
    int ret = std::ranges::max({7, 6, 9, 3, 5, 1, 2, 4}, {}, proj);
    assert(ret == 5);
  }

  { // test comparator
    int ret = std::ranges::max({7, 6, 9, 3, 5, 1, 2, 4}, std::ranges::greater{});
    assert(ret == 1);
  }

  { // check that complexity requirements are met
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
    std::same_as<int> decltype(auto) ret = std::ranges::max({1, 2, 3}, comparator, projection);
    assert(ret == 3);
    assert(compares == 2);
    assert(projections == 4);
  }

  { // check that std::invoke is used
    struct S { int i; };
    std::same_as<S> decltype(auto) ret = std::ranges::max({ S{2}, S{1}, S{3} }, {}, &S::i);
    assert(ret.i == 3);
  }

  { // check that the first largest element is returned
    { // where the first element is the largest
      struct S { int check; int other; };
      auto ret = std::ranges::max({ S{1, 1}, S{0, 2}, S{1, 3} }, {}, &S::check);
      assert(ret.check == 1);
      assert(ret.other == 1);
    }
    { // where the first element isn't the largest
      struct S { int check; int other; };
      auto ret = std::ranges::max({ S{0, 1}, S{1, 2}, S{1, 3} }, {}, &S::check);
      assert(ret.check == 1);
      assert(ret.other == 2);
    }
  }
}

template <class It, class Sent = It>
constexpr void test_range_types() {
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  auto range = std::ranges::subrange(It(a), Sent(It(a + 8)));
  int ret = std::ranges::max(range);
  assert(ret == 9);
}

constexpr void test_range() {
  { // check that all range types work
    test_range_types<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
    test_range_types<forward_iterator<int*>>();
    test_range_types<bidirectional_iterator<int*>>();
    test_range_types<random_access_iterator<int*>>();
    test_range_types<contiguous_iterator<int*>>();
  }

  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  { // test projection
    auto proj = [](int& i) { return i == 5 ? 100 : i; };
    int ret = std::ranges::max(a, std::ranges::less{}, proj);
    assert(ret == 5);
  }

  { // test comparator
    int ret = std::ranges::max(a, std::ranges::greater{});
    assert(ret == 1);
  }

  { // check that predicate and projection call counts are correct
    int compares = 0;
    int projections = 0;
    auto comparator = [&](int x, int y) {
      ++compares;
      return x < y;
    };
    auto projection = [&](int x) {
      ++projections;
      return x;
    };
    std::same_as<int> decltype(auto) ret = std::ranges::max(std::array{1, 2, 3}, comparator, projection);
    assert(ret == 3);
    assert(compares == 2);
    assert(projections == 4);
  }

  { // check that std::invoke is used
    struct S { int i; };
    S b[3] = { S{2}, S{1}, S{3} };
    std::same_as<S> decltype(auto) ret = std::ranges::max(b, {}, &S::i);
    assert(ret.i == 3);
  }

  { // check that the first largest element is returned
    { // where the first element is the largest
      struct S { int check; int other; };
      S b[] = { S{1, 1}, S{0, 2}, S{1, 3} };
      auto ret = std::ranges::max(b, {}, &S::check);
      assert(ret.check == 1);
      assert(ret.other == 1);
    }
    { // where the first element isn't the largest
      struct S { int check; int other; };
      S b[] = { S{0, 1}, S{1, 2}, S{1, 3} };
      auto ret = std::ranges::max(b, {}, &S::check);
      assert(ret.check == 1);
      assert(ret.other == 2);
    }
  }
}

constexpr bool test() {
  test_2_arguments();
  test_initializer_list();
  test_range();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

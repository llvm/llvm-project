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
//   constexpr const T& ranges::min(const T& a, const T& b, Comp comp = {}, Proj proj = {});
// template<copyable T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr T ranges::min(initializer_list<T> r, Comp comp = {}, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   requires indirectly_copyable_storable<iterator_t<R>, range_value_t<R>*>
//   constexpr range_value_t<R>
//     ranges::min(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T>
concept HasMinR = requires { std::ranges::min(std::declval<T>()); };

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

static_assert(!HasMinR<int>);

static_assert(HasMinR<int(&)[10]>);
static_assert(HasMinR<int(&&)[10]>);
static_assert(!HasMinR<NoLessThanOp(&)[10]>);
static_assert(!HasMinR<NotTotallyOrdered(&)[10]>);
static_assert(!HasMinR<Movable(&)[10]>);

static_assert(HasMinR<std::initializer_list<int>>);
static_assert(!HasMinR<std::initializer_list<NoLessThanOp>>);
static_assert(!HasMinR<std::initializer_list<NotTotallyOrdered>>);
static_assert(!HasMinR<std::initializer_list<Movable>>);
static_assert(!HasMinR<InputRangeNotDerivedFrom>);
static_assert(!HasMinR<InputRangeNotIndirectlyReadable>);
static_assert(!HasMinR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasMinR<InputRangeNotSentinelSemiregular>);
static_assert(!HasMinR<InputRangeNotSentinelEqualityComparableWith>);

template <class T, class U = T>
concept HasMin2 = requires { std::ranges::min(std::declval<T>(), std::declval<U>()); };

static_assert(HasMin2<int>);
static_assert(!HasMin2<int, long>);

static_assert(std::is_same_v<decltype(std::ranges::min(1, 2)), const int&>);

constexpr void test_2_arguments() {
  assert(std::ranges::min(1, 2) == 1);
  assert(std::ranges::min(2, 1) == 1);
  // test comparator
  assert(std::ranges::min(1, 2, std::ranges::greater{}) == 2);
  // test projection
  assert(std::ranges::min(1, 2, std::ranges::less{}, [](int i){ return i == 1 ? 10 : i; }) == 2);

  { // check that std::invoke is used
    struct S { int i; };
    S a[3] = { S{2}, S{1}, S{3} };
    decltype(auto) ret = std::ranges::min(a[0], a[1], {}, &S::i);
    ASSERT_SAME_TYPE(decltype(ret), const S&);
    assert(&ret == &a[1]);
    assert(ret.i == 1);
  }

  { // check that pointers are compared and not a range
    int i;
    int* a[] = {&i, &i + 1};
    auto ret = std::ranges::min(a[0], a[1]);
    assert(ret == &i);
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
    auto ret = std::ranges::min(1, 2, comparator, projection);
    assert(ret == 1);
    assert(compares == 1);
    assert(projections == 2);
  }

  { // check that the first argument is returned
    struct S { int check; int other; };
    auto ret = std::ranges::min(S {0, 1}, S {0, 2}, {}, &S::check);
    assert(ret.other == 1);
  }
}

constexpr void test_initializer_list() {
  {
    // test projection
    auto proj = [](int i) { return i == 5 ? -100 : i; };
    int ret = std::ranges::min({7, 6, 9, 3, 5, 1, 2, 4}, std::ranges::less{}, proj);
    assert(ret == 5);
  }

  {
    // test comparator
    int ret = std::ranges::min({7, 6, 9, 3, 5, 1, 2, 4}, std::ranges::greater{});
    assert(ret == 9);
  }

  {
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
    std::same_as<int> decltype(auto) ret = std::ranges::min({1, 2, 3}, comparator, projection);
    assert(ret == 1);
    assert(compares == 2);
    assert(projections == 4);
  }

  {
    struct S { int i; };
    decltype(auto) ret = std::ranges::min({ S{2}, S{1}, S{3} }, {}, &S::i);
    ASSERT_SAME_TYPE(decltype(ret), S);
    assert(ret.i == 1);
  }

  {
    int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
    using It = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<It>;
    auto range = std::ranges::subrange(It(a), Sent(It(a + 8)));
    auto ret = std::ranges::min(range);
    assert(ret == 1);
  }
}

template <class It, class Sent = It>
constexpr void test_range_types() {
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  auto range = std::ranges::subrange(It(a), Sent(It(a + 8)));
  int ret = std::ranges::min(range);
  assert(ret == 1);
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
  {
    // test projection
    auto proj = [](int& i) { return i == 5 ? -100 : i; };
    int ret = std::ranges::min(a, std::ranges::less{}, proj);
    assert(ret == 5);
  }

  {
    // test comparator
    int ret = std::ranges::min(a, std::ranges::greater{});
    assert(ret == 9);
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
    std::same_as<int> decltype(auto) ret = std::ranges::min(std::array{1, 2, 3}, comparator, projection);
    assert(ret == 1);
    assert(compares == 2);
    assert(projections == 4);
  }

  { // check that std::invoke is used
    struct S { int i; };
    S b[3] = { S{2}, S{1}, S{3} };
    std::same_as<S> decltype(auto) ret = std::ranges::min(b, {}, &S::i);
    assert(ret.i == 1);
  }

  { // check that the first smallest element is returned
    { // where the first element is the smallest
      struct S { int check; int other; };
      S b[] = { S{0, 1}, S{1, 2}, S{0, 3} };
      auto ret = std::ranges::min(b, {}, &S::check);
      assert(ret.check == 0);
      assert(ret.other == 1);
    }
    { // where the first element isn't the smallest
      struct S { int check; int other; };
      S b[] = { S{2, 1}, S{0, 2}, S{0, 3} };
      auto ret = std::ranges::min(b, {}, &S::check);
      assert(ret.check == 0);
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

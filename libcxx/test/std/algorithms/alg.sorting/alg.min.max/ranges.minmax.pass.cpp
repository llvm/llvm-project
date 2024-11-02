//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr ranges::minmax_result<const T&>
//     ranges::minmax(const T& a, const T& b, Comp comp = {}, Proj proj = {});
// template<copyable T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr ranges::minmax_result<T>
//     ranges::minmax(initializer_list<T> r, Comp comp = {}, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//   requires indirectly_copyable_storable<iterator_t<R>, range_value_t<R>*>
//   constexpr ranges::minmax_result<range_value_t<R>>
//     ranges::minmax(R&& r, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <ranges>

#include "test_iterators.h"

template <class T>
concept HasMinMax = requires { std::ranges::minmax(std::declval<T>()); };

struct NoLessThanOp {};
struct NotTotallyOrdered {
  int i;
  bool operator<(const NotTotallyOrdered& o) const { return i < o.i; }
};
struct MoveOnly {
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly(const MoveOnly&) = delete;
};

static_assert(!HasMinMax<int>);

static_assert(HasMinMax<int (&)[10]>); // make sure HasMinMax works with an array
static_assert(!HasMinMax<NoLessThanOp (&)[10]>);
static_assert(!HasMinMax<NotTotallyOrdered (&)[10]>);
static_assert(!HasMinMax<MoveOnly (&)[10]>);

static_assert(HasMinMax<std::initializer_list<int>>); // make sure HasMinMax works with an initializer_list
static_assert(!HasMinMax<std::initializer_list<NoLessThanOp>>);
static_assert(!HasMinMax<std::initializer_list<NotTotallyOrdered>>);
static_assert(!HasMinMax<std::initializer_list<MoveOnly>>);

static_assert(std::is_same_v<std::ranges::minmax_result<int>, std::ranges::min_max_result<int>>);

static_assert(std::is_same_v<decltype(std::ranges::minmax(1, 2)), std::ranges::minmax_result<const int&>>);

constexpr void test_2_arguments() {
  const int one = 1;
  const int two = 2;
  {
    auto result = std::ranges::minmax(one, two);
    assert(result.min == 1);
    assert(result.max == 2);
  }

  {
    auto result = std::ranges::minmax(two, one);
    assert(result.min == 1);
    assert(result.max == 2);
  }

  {
    // test comparator
    auto result = std::ranges::minmax(one, two, std::ranges::greater{});
    assert(result.min == 2);
    assert(result.max == 1);
  }

  {
    // test projection
    auto result = std::ranges::minmax(one, two, std::ranges::less{}, [](int i) { return i == 1 ? 10 : i; });
    assert(result.min == 2);
    assert(result.max == 1);
  }

  {
    // test if std::invoke is used for the predicate
    struct S {
      int i;
    };
    S a[3] = {S{2}, S{1}, S{3}};
    std::same_as<std::ranges::minmax_result<const S&>> auto ret = std::ranges::minmax(a[0], a[1], {}, &S::i);
    assert(&ret.min == &a[1]);
    assert(&ret.max == &a[0]);
    assert(ret.min.i == 1);
    assert(ret.max.i == 2);
  }

  {
    // check that std::invoke is used for the comparator
    struct S {
      int i;
      constexpr bool comp(S rhs) const { return i < rhs.i; }
    };
    S a[] = {{2}, {5}};
    auto ret = std::ranges::minmax(a[0], a[1], &S::comp);
    assert(ret.min.i == 2);
    assert(ret.max.i == 5);
  }

  {
    // make sure that {a, b} is returned if b is not less than a
    auto r = std::ranges::minmax(one, two);
    assert(&r.min == &one);
    assert(&r.max == &two);
  }
}

constexpr void test_initializer_list() {
  {
    // test projection
    auto proj = [](int i) { return i == 5 ? -100 : i; };
    auto ret = std::ranges::minmax({7, 6, 9, 3, 5, 1, 2, 4}, std::ranges::less{}, proj);
    assert(ret.min == 5);
    assert(ret.max == 9);
  }

  {
    // test comparator
    auto ret = std::ranges::minmax({7, 6, 9, 3, 5, 1, 2, 4}, std::ranges::greater{});
    assert(ret.min == 9);
    assert(ret.max == 1);
  }

  {
    // check predicate and projection call counts
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
    std::same_as<std::ranges::minmax_result<int>> auto ret = std::ranges::minmax({1, 2, 3}, comparator, projection);
    assert(ret.min == 1);
    assert(ret.max == 3);
    assert(compares == 3);
    assert(projections == 6);
  }

  {
    // check that std::invoke is used for the predicate
    struct S {
      int i;
    };
    std::same_as<std::ranges::minmax_result<S>> auto ret = std::ranges::minmax({S{2}, S{1}, S{3}}, {}, &S::i);
    assert(ret.min.i == 1);
    assert(ret.max.i == 3);
  }

  {
    // check that std::invoke is used for the comparator
    struct S {
      int i;
      constexpr bool comp(S rhs) const { return i < rhs.i; }
    };
    auto ret = std::ranges::minmax({S {1}, S {2}, S {3}, S {4}}, &S::comp);
    assert(ret.min.i == 1);
    assert(ret.max.i == 4);
  }

  {
    // check that a single element works
    auto ret = std::ranges::minmax({ 1 });
    assert(ret.min == 1);
    assert(ret.max == 1);
  }

  {
    // check in ascending order
    auto ret = std::ranges::minmax({1, 2, 3});
    assert(ret.min == 1);
    assert(ret.max == 3);
  }

  {
    // check in descending order
    auto ret = std::ranges::minmax({3, 2, 1});
    assert(ret.min == 1);
    assert(ret.max == 3);
  }
}

template <class Iter, class Sent = Iter>
constexpr void test_range_types() {
  {
    // test projection
    int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
    auto proj = [](int& i) { return i == 5 ? -100 : i; };
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 8)));
    auto ret = std::ranges::minmax(range, std::ranges::less{}, proj);
    assert(ret.min == 5);
    assert(ret.max == 9);
  }

  {
    // test comparator
    int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 8)));
    auto ret = std::ranges::minmax(range, std::ranges::greater{});
    assert(ret.min == 9);
    assert(ret.max == 1);
  }

  {
    // check that complexity requirements are met
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
    int a[] = {1, 2, 3};
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
    std::same_as<std::ranges::minmax_result<int>> auto ret = std::ranges::minmax(range, comparator, projection);
    assert(ret.min == 1);
    assert(ret.max == 3);
    assert(compares == 3);
    assert(projections == 6);
  }

  {
    // check that a single element works
    int a[] = { 1 };
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 1)));
    auto ret = std::ranges::minmax(range);
    assert(ret.min == 1);
    assert(ret.max == 1);
  }

  {
    // check in ascending order
    int a[] = {1, 2, 3};
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
    auto ret = std::ranges::minmax(range);
    assert(ret.min == 1);
    assert(ret.max == 3);
  }

  {
    // check in descending order
    int a[] = {3, 2, 1};
    auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
    auto ret = std::ranges::minmax(range);
    assert(ret.min == 1);
    assert(ret.max == 3);
  }
}

constexpr void test_range() {
  test_range_types<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_range_types<forward_iterator<int*>>();
  test_range_types<bidirectional_iterator<int*>>();
  test_range_types<random_access_iterator<int*>>();
  test_range_types<contiguous_iterator<int*>>();

  {
    // check that std::invoke is used for the predicate
    struct S {
      int i;
    };
    S b[3] = {S{2}, S{1}, S{3}};
    std::same_as<std::ranges::minmax_result<S>> auto ret = std::ranges::minmax(b, {}, &S::i);
    assert(ret.min.i == 1);
    assert(ret.max.i == 3);
  }

  {
    // check that std::invoke is used for the comparator
    struct S {
      int i;
      constexpr bool comp(S rhs) const { return i < rhs.i; }
    };
    S a[] = {{1}, {2}, {3}, {4}};
    auto ret = std::ranges::minmax(a, &S::comp);
    assert(ret.min.i == 1);
    assert(ret.max.i == 4);
  }

  {
    // check that the leftmost value for min an rightmost for max are returned
    struct S {
      int comp;
      int other;
    };
    S a[] = { {0, 0}, {0, 2}, {0, 1} };
    auto ret = std::ranges::minmax(a, {}, &S::comp);
    assert(ret.min.comp == 0);
    assert(ret.max.comp == 0);
    assert(ret.min.other == 0);
    assert(ret.max.other == 1);
  }

  {
    // check that an rvalue array works
    int a[] = {1, 2, 3, 4};
    auto ret = std::ranges::minmax(std::move(a));
    assert(ret.min == 1);
    assert(ret.max == 4);
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

  {
    // check that the iterator isn't moved from multiple times
    std::shared_ptr<int> a[] = { std::make_shared<int>(42) };
    auto range = std::ranges::subrange(std::move_iterator(a), std::move_iterator(a + 1));
    auto [min, max] = std::ranges::minmax(range);
    assert(a[0] == nullptr);
    assert(min != nullptr);
    assert(max == min);
  }

  return 0;
}

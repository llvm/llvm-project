//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// move_iterator

// template <class Iter1, three_way_comparable_with<Iter1> Iter2>
//   constexpr auto operator<=>(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y)
//     -> compare_three_way_result_t<Iter1, Iter2>;

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"


template<class T, class U> concept HasEquals = requires (T t, U u) { t == u; };
template<class T, class U> concept HasSpaceship = requires (T t, U u) { t <=> u; };

static_assert(!HasSpaceship<std::move_iterator<int*>, std::move_iterator<char*>>);
static_assert( HasSpaceship<std::move_iterator<int*>, std::move_iterator<int*>>);
static_assert( HasSpaceship<std::move_iterator<int*>, std::move_iterator<const int*>>);
static_assert( HasSpaceship<std::move_iterator<const int*>, std::move_iterator<const int*>>);
static_assert(!HasSpaceship<std::move_iterator<forward_iterator<int*>>, std::move_iterator<forward_iterator<int*>>>);
static_assert(!HasSpaceship<std::move_iterator<random_access_iterator<int*>>, std::move_iterator<random_access_iterator<int*>>>);
static_assert(!HasSpaceship<std::move_iterator<contiguous_iterator<int*>>, std::move_iterator<contiguous_iterator<int*>>>);
static_assert( HasSpaceship<std::move_iterator<three_way_contiguous_iterator<int*>>, std::move_iterator<three_way_contiguous_iterator<int*>>>);

static_assert(!HasSpaceship<std::move_iterator<int*>, std::move_sentinel<int*>>);
static_assert(!HasSpaceship<std::move_iterator<three_way_contiguous_iterator<int*>>, std::move_sentinel<three_way_contiguous_iterator<int*>>>);

void test_spaceshippable_but_not_three_way_comparable() {
  struct A {
    using value_type = int;
    using difference_type = int;
    int& operator*() const;
    A& operator++();
    A operator++(int);
    std::strong_ordering operator<=>(const A&) const;
  };
  struct B {
    using value_type = int;
    using difference_type = int;
    int& operator*() const;
    B& operator++();
    B operator++(int);
    std::strong_ordering operator<=>(const B&) const;
    bool operator==(const A&) const;
    std::strong_ordering operator<=>(const A&) const;
  };
  static_assert( std::input_iterator<A>);
  static_assert( std::input_iterator<B>);
  static_assert( HasEquals<A, B>);
  static_assert( HasSpaceship<A, B>);
  static_assert(!std::three_way_comparable_with<A, B>);
  static_assert( HasEquals<std::move_iterator<A>, std::move_iterator<B>>);
  static_assert(!HasSpaceship<std::move_iterator<A>, std::move_iterator<B>>);
}

template <class It, class Jt>
constexpr void test_two()
{
  int a[] = {3, 1, 4};
  const std::move_iterator<It> i1 = std::move_iterator<It>(It(a));
  const std::move_iterator<It> i2 = std::move_iterator<It>(It(a + 2));
  const std::move_iterator<Jt> j1 = std::move_iterator<Jt>(Jt(a));
  const std::move_iterator<Jt> j2 = std::move_iterator<Jt>(Jt(a + 2));
  ASSERT_SAME_TYPE(decltype(i1 <=> i2), std::strong_ordering);
  assert((i1 <=> i1) == std::strong_ordering::equal);
  assert((i1 <=> i2) == std::strong_ordering::less);
  assert((i2 <=> i1) == std::strong_ordering::greater);
  ASSERT_SAME_TYPE(decltype(i1 <=> j2), std::strong_ordering);
  assert((i1 <=> j1) == std::strong_ordering::equal);
  assert((i1 <=> j2) == std::strong_ordering::less);
  assert((i2 <=> j1) == std::strong_ordering::greater);
}

constexpr bool test()
{
  test_two<int*, int*>();
  test_two<int*, const int*>();
  test_two<const int*, int*>();
  test_two<const int*, const int*>();
  test_two<three_way_contiguous_iterator<int*>, three_way_contiguous_iterator<int*>>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());

  return 0;
}

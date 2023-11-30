//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2>
//   requires indirectly_swappable<I1, I2>
//   constexpr ranges::swap_ranges_result<I1, I2>
//     ranges::swap_ranges(I1 first1, S1 last1, I2 first2, S2 last2);
// template<input_range R1, input_range R2>
//   requires indirectly_swappable<iterator_t<R1>, iterator_t<R2>>
//   constexpr ranges::swap_ranges_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>>
//     ranges::swap_ranges(R1&& r1, R2&& r2);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "test_iterators.h"

constexpr void test_different_lengths() {
  using Expected = std::ranges::swap_ranges_result<int*, int*>;
  int i[3] = {1, 2, 3};
  int j[1] = {4};
  std::same_as<Expected> auto r = std::ranges::swap_ranges(i, i + 3, j, j + 1);
  assert(r.in1 == i + 1);
  assert(r.in2 == j + 1);
  assert(i[0] == 4);
  assert(i[1] == 2);
  assert(i[2] == 3);
  assert(j[0] == 1);
  std::same_as<Expected> auto r2 = std::ranges::swap_ranges(i, j);
  assert(r2.in1 == i + 1);
  assert(r2.in2 == j + 1);
  assert(i[0] == 1);
  assert(i[1] == 2);
  assert(i[2] == 3);
  assert(j[0] == 4);
  std::same_as<Expected> auto r3 = std::ranges::swap_ranges(j, j + 1, i, i + 3);
  assert(r3.in1 == j + 1);
  assert(r3.in2 == i + 1);
  assert(i[0] == 4);
  assert(i[1] == 2);
  assert(i[2] == 3);
  assert(j[0] == 1);
  std::same_as<Expected> auto r4 = std::ranges::swap_ranges(j, i);
  assert(r4.in1 == j + 1);
  assert(r4.in2 == i + 1);
  assert(i[0] == 1);
  assert(i[1] == 2);
  assert(i[2] == 3);
  assert(j[0] == 4);
}

constexpr void test_range() {
  std::array r1 = {1, 2, 3};
  std::array r2 = {4, 5, 6};


  std::same_as<std::ranges::in_in_result<std::array<int, 3>::iterator, std::array<int, 3>::iterator>> auto r = std::ranges::swap_ranges(r1, r2);
  assert(r.in1 == r1.end());
  assert(r.in2 == r2.end());

  assert((r1 == std::array{4, 5, 6}));
  assert((r2 == std::array{1, 2, 3}));
}

constexpr void test_borrowed_input_range() {
  {
    int r1[] = {1, 2, 3};
    int r2[] = {4, 5, 6};
    std::ranges::swap_ranges(std::views::all(r1), r2);
    assert(r1[0] == 4);
    assert(r1[1] == 5);
    assert(r1[2] == 6);
    assert(r2[0] == 1);
    assert(r2[1] == 2);
    assert(r2[2] == 3);
  }
  {
    int r1[] = {1, 2, 3};
    int r2[] = {4, 5, 6};
    std::ranges::swap_ranges(r1, std::views::all(r2));
    assert(r1[0] == 4);
    assert(r1[1] == 5);
    assert(r1[2] == 6);
    assert(r2[0] == 1);
    assert(r2[1] == 2);
    assert(r2[2] == 3);
  }
  {
    int r1[] = {1, 2, 3};
    int r2[] = {4, 5, 6};
    std::ranges::swap_ranges(std::views::all(r1), std::views::all(r2));
    assert(r1[0] == 4);
    assert(r1[1] == 5);
    assert(r1[2] == 6);
    assert(r2[0] == 1);
    assert(r2[1] == 2);
    assert(r2[2] == 3);
  }
}

constexpr void test_sentinel() {
  int i[3] = {1, 2, 3};
  int j[3] = {4, 5, 6};
  using It = cpp17_input_iterator<int*>;
  using Sent = sentinel_wrapper<It>;
  using Expected = std::ranges::swap_ranges_result<It, It>;
  std::same_as<Expected> auto r =
      std::ranges::swap_ranges(It(i), Sent(It(i + 3)), It(j), Sent(It(j + 3)));
  assert(base(r.in1) == i + 3);
  assert(base(r.in2) == j + 3);
  assert(i[0] == 4);
  assert(i[1] == 5);
  assert(i[2] == 6);
  assert(j[0] == 1);
  assert(j[1] == 2);
  assert(j[2] == 3);
}

template <class Iter1, class Iter2>
constexpr void test_iterators() {
  using Expected = std::ranges::swap_ranges_result<Iter1, Iter2>;
  int i[3] = {1, 2, 3};
  int j[3] = {4, 5, 6};
  std::same_as<Expected> auto r =
      std::ranges::swap_ranges(Iter1(i), sentinel_wrapper(Iter1(i + 3)), Iter2(j), sentinel_wrapper(Iter2(j + 3)));
  assert(base(r.in1) == i + 3);
  assert(base(r.in2) == j + 3);
  assert(i[0] == 4);
  assert(i[1] == 5);
  assert(i[2] == 6);
  assert(j[0] == 1);
  assert(j[1] == 2);
  assert(j[2] == 3);
}

constexpr void test_rval_range() {
  {
    using Expected = std::ranges::swap_ranges_result<std::array<int, 3>::iterator, std::ranges::dangling>;
    std::array<int, 3> r = {1, 2, 3};
    std::same_as<Expected> auto a = std::ranges::swap_ranges(r, std::array{4, 5, 6});
    assert((r == std::array{4, 5, 6}));
    assert(a.in1 == r.begin() + 3);
  }
  {
    std::array<int, 3> r = {1, 2, 3};
    using Expected = std::ranges::swap_ranges_result<std::ranges::dangling, std::array<int, 3>::iterator>;
    std::same_as<Expected> auto b = std::ranges::swap_ranges(std::array{4, 5, 6}, r);
    assert((r == std::array{4, 5, 6}));
    assert(b.in2 == r.begin() + 3);
  }
}

template <class Out>
constexpr void test_proxy_in_iterators() {
  test_iterators<ProxyIterator<cpp20_input_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<forward_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<bidirectional_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<random_access_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<contiguous_iterator<int*>>, Out>();
}

constexpr bool test() {
  test_range();

  test_iterators<cpp20_input_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iterators<cpp20_input_iterator<int*>, forward_iterator<int*>>();
  test_iterators<cpp20_input_iterator<int*>, bidirectional_iterator<int*>>();
  test_iterators<cpp20_input_iterator<int*>, random_access_iterator<int*>>();
  test_iterators<cpp20_input_iterator<int*>, int*>();

  test_iterators<forward_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iterators<forward_iterator<int*>, forward_iterator<int*>>();
  test_iterators<forward_iterator<int*>, bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>, random_access_iterator<int*>>();
  test_iterators<forward_iterator<int*>, int*>();

  test_iterators<bidirectional_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, bidirectional_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, random_access_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, int*>();

  test_iterators<random_access_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, random_access_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, int*>();

  test_iterators<int*, cpp20_input_iterator<int*>>();
  test_iterators<int*, forward_iterator<int*>>();
  test_iterators<int*, bidirectional_iterator<int*>>();
  test_iterators<int*, random_access_iterator<int*>>();
  test_iterators<int*, int*>();

  test_proxy_in_iterators<ProxyIterator<cpp20_input_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<forward_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<bidirectional_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<random_access_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<contiguous_iterator<int*>>>();

  test_sentinel();
  test_different_lengths();
  test_borrowed_input_range();
  test_rval_range();

  return true;
}

static_assert(std::same_as<std::ranges::swap_ranges_result<int, char>, std::ranges::in_in_result<int, char>>);

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<InnerView>;
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const InnerView>;

#include <cassert>
#include <cstddef>
#include <ranges>
#include <utility>

#include "helpers.h"
#include "test_iterators.h"
#include "test_macros.h"

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

struct NonApproximatelySizedView : std::ranges::view_base {
  using iterator = forward_iterator<int*>;
  iterator begin() const;
  iterator end() const;
};

struct ApproximatelySizedView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedView(unsigned int size) : size_(size) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + size_); }
  constexpr unsigned int reserve_hint() const { return size_; }
};

struct ApproximatelySizedNotConstView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedNotConstView(unsigned int size) : size_(size) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + size_); }
  constexpr unsigned int reserve_hint() { return size_; }
};

// Test with different values of N for an approximately sized view
template <std::size_t N, class Fn>
constexpr void test_approx_sized_view() {
  std::ranges::adjacent_transform_view<ApproximatelySizedView, Fn, N> v(ApproximatelySizedView(5), Fn{});
  static_assert(std::ranges::approximately_sized_range<decltype(v)>);
  static_assert(std::ranges::approximately_sized_range<const decltype(v)>);

  auto expected_hint = 5 - (N - 1);
  assert(v.reserve_hint() == expected_hint);
  assert(std::as_const(v).reserve_hint() == expected_hint);
}

// Test with different values of N for a non-const approximately sized view
template <std::size_t N, class Fn>
constexpr void test_nonconst_approx_sized() {
  // non-const-only reserve_hint
  std::ranges::adjacent_transform_view<ApproximatelySizedNotConstView, Fn, N> v(
      ApproximatelySizedNotConstView(5), Fn{});
  static_assert(std::ranges::approximately_sized_range<decltype(v)>);
  static_assert(!std::ranges::approximately_sized_range<const decltype(v)>);

  auto expected_hint = 5 - (N - 1);
  assert(v.reserve_hint() == expected_hint);
}

template <std::size_t N, class Fn>
constexpr void test_empty_range() {
  std::ranges::adjacent_transform_view<ApproximatelySizedView, Fn, N> v(ApproximatelySizedView(0), Fn{});
  static_assert(std::ranges::approximately_sized_range<decltype(v)>);
  static_assert(std::ranges::approximately_sized_range<const decltype(v)>);

  assert(v.reserve_hint() == 0);
  assert(std::as_const(v).reserve_hint() == 0);
}

template <std::size_t N, class Fn>
constexpr void test_N_greater_than_size() {
  if constexpr (N > 2) {
    std::ranges::adjacent_transform_view<ApproximatelySizedView, Fn, N> v(ApproximatelySizedView(2), Fn{});
    static_assert(std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(std::ranges::approximately_sized_range<const decltype(v)>);
    assert(v.reserve_hint() == 0);
    assert(std::as_const(v).reserve_hint() == 0);
  }
}

template <std::size_t N, class Fn>
constexpr void test() {
  test_approx_sized_view<N, Fn>();
  test_nonconst_approx_sized<N, Fn>();
  test_empty_range<N, Fn>();
  test_N_greater_than_size<N, Fn>();
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();
}

constexpr bool test() {
  // non-approximately-sized range has no reserve_hint
  static_assert(!std::ranges::approximately_sized_range<
                std::ranges::adjacent_transform_view<NonApproximatelySizedView, Multiply, 2>>);
  static_assert(!std::ranges::approximately_sized_range<
                const std::ranges::adjacent_transform_view<NonApproximatelySizedView, Multiply, 2>>);

  test<1>();
  test<2>();
  test<3>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size() requires sized_range<InnerView>
// constexpr auto size() const requires sized_range<const InnerView>

#include <ranges>

#include "test_macros.h"
#include "helpers.h"
#include "../range_adaptor_types.h"

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
struct View : std::ranges::view_base {
  std::size_t size_ = 0;
  constexpr View(std::size_t s) : size_(s) {}
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + size_; }
};

struct SizedNonConst : std::ranges::view_base {
  using iterator    = forward_iterator<int*>;
  std::size_t size_ = 0;
  constexpr SizedNonConst(std::size_t s) : size_(s) {}
  constexpr auto begin() const { return iterator{buffer}; }
  constexpr auto end() const { return iterator{buffer + size_}; }
  constexpr std::size_t size() { return size_; }
};

struct StrangeSizeView : std::ranges::view_base {
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + 8; }

  constexpr auto size() { return 5; }
  constexpr auto size() const { return 6; }
};

// Test with different values of N for a sized view
template <std::size_t N, class Fn>
constexpr void test_sized_view() {
  std::ranges::adjacent_transform_view<View, Fn, N> v(View(8), Fn{});
  static_assert(std::ranges::sized_range<decltype(v)>);
  static_assert(std::ranges::sized_range<const decltype(v)>);

  auto expected_size = 8 - (N - 1);
  assert(v.size() == expected_size);
  assert(std::as_const(v).size() == expected_size);
}

// Test with different values of N for a non-const sized view
template <std::size_t N, class Fn>
constexpr void test_nonconst_sized() {
  std::ranges::adjacent_transform_view<SizedNonConst, Fn, N> v(SizedNonConst(5), Fn{});
  static_assert(std::ranges::sized_range<decltype(v)>);
  static_assert(!std::ranges::sized_range<const decltype(v)>);

  auto expected_size = 5 - (N - 1);
  assert(v.size() == expected_size);
}

// Test with different values of N for a view with different const/non-const sizes
template <std::size_t N, class Fn>
constexpr void test_strange_size() {
  std::ranges::adjacent_transform_view<StrangeSizeView, Fn, N> v(StrangeSizeView{}, Fn{});
  static_assert(std::ranges::sized_range<decltype(v)>);
  static_assert(std::ranges::sized_range<const decltype(v)>);

  assert(v.size() == 5 - (N - 1));
  assert(std::as_const(v).size() == 6 - (N - 1));
}

template <std::size_t N, class Fn>
constexpr void test_empty_range() {
  std::ranges::adjacent_transform_view<View, Fn, N> v(View(0), Fn{});
  static_assert(std::ranges::sized_range<decltype(v)>);
  static_assert(std::ranges::sized_range<const decltype(v)>);

  assert(v.size() == 0);
  assert(std::as_const(v).size() == 0);
}

template <std::size_t N, class Fn>
constexpr void test_N_greater_than_size() {
  if constexpr (N > 2) {
    std::ranges::adjacent_transform_view<View, Fn, N> v(View(2), Fn{});
    static_assert(std::ranges::sized_range<decltype(v)>);
    static_assert(std::ranges::sized_range<const decltype(v)>);
    assert(v.size() == 0);
    assert(std::as_const(v).size() == 0);
  }
}

template <std::size_t N, class Fn>
constexpr void test() {
  test_sized_view<N, Fn>();
  test_nonconst_sized<N, Fn>();
  test_strange_size<N, Fn>();
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

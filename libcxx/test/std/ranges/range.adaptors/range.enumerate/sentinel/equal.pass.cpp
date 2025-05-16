//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::sentinel

// template<bool OtherConst>
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>

#include "test_iterators.h"

#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = MinimalView<Iterator, Sentinel>;

  std::array array{0, 1, 2, 3, 84};

  View mv{Iterator(std::to_address(base(array.begin()))), Sentinel(Iterator(std::to_address(base(array.end()))))};
  std::ranges::enumerate_view view(std::move(mv));

  auto const it = view.begin();
  auto const s  = view.end();

  std::same_as<bool> decltype(auto) eqItSResult = (it == s);
  assert(!eqItSResult);
  std::same_as<bool> decltype(auto) eqSItResult = (s == it);
  assert(!eqSItResult);

  std::same_as<bool> decltype(auto) neqItSResult = (it != s);
  assert(neqItSResult);
  std::same_as<bool> decltype(auto) neqSItResult = (s != it);
  assert(neqSItResult);
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}

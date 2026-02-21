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

// constexpr sentinel_t<Base> base() const;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <utility>

#include "test_iterators.h"

#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView>;

  std::array<int, 5> array{0, 1, 2, 3, 84};

  View mv{Iterator(std::to_address(base(array.begin()))), Sentinel(Iterator(std::to_address(base(array.end()))))};
  EnumerateView ev{std::move(mv)};

  EnumerateSentinel const s                    = ev.end();
  std::same_as<Sentinel> decltype(auto) result = s.base();
  assert(base(base(result)) == std::to_address(base(array.end())));
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

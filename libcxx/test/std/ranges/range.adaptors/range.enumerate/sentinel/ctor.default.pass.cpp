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

// sentinel() = default;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "../types.h"

struct PODSentinel {
  int i; // deliberately uninitialised

  friend constexpr bool operator==(std::tuple<int>*, const PODSentinel&) { return true; }
};

template <typename Iterator, typename Sentinel>
struct PODSentinelView : MinimalView<Iterator, Sentinel> {
  std::tuple<int>* begin() const;
  PODSentinel end();
};

template <class Iterator>
constexpr void test() {
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = PODSentinelView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView>;

  {
    EnumerateSentinel s;

    assert(s.base().i == 0);
  }

  {
    EnumerateSentinel s = {};

    assert(s.base().i == 0);
  }

  static_assert(noexcept(EnumerateSentinel()));
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

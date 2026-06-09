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

// class enumerate_view::iterator

// constexpr const iterator_t<Base>& base() const & noexcept;
// constexpr iterator_t<Base> base() &&;

#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "test_iterators.h"

#include "../types.h"

template <typename Iter>
constexpr void testBase_with_CommonRange() {
  using CommonRange = MinimalRange<Iter, Iter>;
  static_assert(std::ranges::range<CommonRange>);
  static_assert(std::ranges::common_range<CommonRange>);

  int arr[] = {94, 1, 2, 82};

  CommonRange range{Iter{arr}, Iter{arr + 4}};

  auto ev = range | std::views::enumerate;

  auto it = ev.begin();

  // Test the const& version
  {
    std::same_as<const Iter&> decltype(auto) resultIt = std::as_const(it).base();
    static_assert(noexcept((it.base())));

    assert(base(resultIt) == &arr[0]);
    assert(*resultIt == 94);
  }

  // Test the && version
  {
    std::same_as<Iter> decltype(auto) resultIt = std::move(it).base();

    assert(base(resultIt) == &arr[0]);
    assert(*resultIt == 94);
  }
}

template <typename Iter>
constexpr void testBase_with_NonCommonRange() {
  using Sent           = sentinel_wrapper<Iter>;
  using NonCommonRange = MinimalRange<Iter, Sent>;
  static_assert(!std::ranges::common_range<NonCommonRange>);

  int arr[] = {94, 1, 2, 82};

  NonCommonRange range{Iter{arr}, Sent{Iter{arr + 4}}};

  auto ev = range | std::views::enumerate;

  auto it = ev.begin();

  // Test the const& version
  {
    std::same_as<const Iter&> decltype(auto) resultIt = std::as_const(it).base();
    static_assert(noexcept((it.base())));

    assert(base(resultIt) == &arr[0]);
    assert(*resultIt == 94);
  }

  // Test the && version
  {
    std::same_as<Iter> decltype(auto) resultIt = std::move(it).base();

    assert(base(resultIt) == &arr[0]);
    assert(*resultIt == 94);
  }
}

constexpr bool test() {
  testBase_with_CommonRange<forward_iterator<int*>>();
  testBase_with_CommonRange<bidirectional_iterator<int*>>();
  testBase_with_CommonRange<random_access_iterator<int*>>();
  testBase_with_CommonRange<contiguous_iterator<int*>>();
  testBase_with_CommonRange<int*>();
  testBase_with_CommonRange<int const*>();

  testBase_with_NonCommonRange<cpp17_input_iterator<int*>>();
  testBase_with_NonCommonRange<cpp20_input_iterator<int*>>();
  testBase_with_NonCommonRange<forward_iterator<int*>>();
  testBase_with_NonCommonRange<bidirectional_iterator<int*>>();
  testBase_with_NonCommonRange<random_access_iterator<int*>>();
  testBase_with_NonCommonRange<contiguous_iterator<int*>>();
  testBase_with_NonCommonRange<int*>();
  testBase_with_NonCommonRange<int const*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

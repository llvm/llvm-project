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

// iterator() requires default_initializable<iterator_t<Base>>;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "../types.h"

template <class Iterator, bool IsNoexcept = true>
constexpr void test_default_constructible() {
  using View              = MinimalView<Iterator, sentinel_wrapper<Iterator>>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  EnumerateIterator it1;
  EnumerateIterator it2{};

  assert(it1 == it2);

  static_assert(noexcept(EnumerateIterator()) == IsNoexcept);
}

template <class Iterator>
constexpr void test_not_default_constructible() {
  // Make sure the iterator is *not* default constructible when the underlying iterator isn't.
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  static_assert(!std::is_default_constructible_v<EnumerateIterator>);
}

constexpr bool tests() {
  // clang-format off
  test_not_default_constructible<cpp17_input_iterator<int*>>();
  test_not_default_constructible<cpp20_input_iterator<int*>>();
  test_default_constructible<forward_iterator<int*>,       /* noexcept */ false>();
  test_default_constructible<bidirectional_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<random_access_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<contiguous_iterator<int*>,    /* noexcept */ false>();
  test_default_constructible<int*,                         /* noexcept */ true>();
  // clang-format on

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<I, Proj>> Comp = ranges::less>
//   constexpr subrange<I>
//     equal_range(I first, S last, const T& value, Comp comp = {}, Proj proj = {});                // Since C++20
//
// template<forward_range R, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<iterator_t<R>, Proj>> Comp =
//            ranges::less>
//   constexpr borrowed_subrange_t<R>
//     equal_range(R&& r, const T& value, Comp comp = {}, Proj proj = {});                          // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

// TODO: SFINAE tests.

constexpr bool test() {
  // TODO: main tests.
  // TODO: A custom comparator works.
  // TODO: A custom projection works.

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

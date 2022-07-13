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

// template<bidirectional_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//          class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   I inplace_merge(I first, I middle, S last, Comp comp = {}, Proj proj = {});                    // Since C++20
//
// template<bidirectional_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   borrowed_iterator_t<R>
//     inplace_merge(R&& r, iterator_t<R> middle, Comp comp = {},
//                   Proj proj = {});                                                               // Since C++20

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

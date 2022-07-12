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

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Proj1 = identity, class Proj2 = identity,
//          indirect_strict_weak_order<projected<I1, Proj1>, projected<I2, Proj2>> Comp =
//            ranges::less>
//   constexpr bool includes(I1 first1, S1 last1, I2 first2, S2 last2, Comp comp = {},
//                           Proj1 proj1 = {}, Proj2 proj2 = {});                                   // Since C++20
//
// template<input_range R1, input_range R2, class Proj1 = identity,
//          class Proj2 = identity,
//          indirect_strict_weak_order<projected<iterator_t<R1>, Proj1>,
//                                     projected<iterator_t<R2>, Proj2>> Comp = ranges::less>
//   constexpr bool includes(R1&& r1, R2&& r2, Comp comp = {},
//                           Proj1 proj1 = {}, Proj2 proj2 = {});                                   // Since C++20

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

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

// template<input_iterator I1, sentinel_for<I1> S1,
//              random_access_iterator I2, sentinel_for<I2> S2,
//              class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//       requires indirectly_copyable<I1, I2> && sortable<I2, Comp, Proj2> &&
//                indirect_strict_weak_order<Comp, projected<I1, Proj1>, projected<I2, Proj2>>
//       constexpr partial_sort_copy_result<I1, I2>
//         partial_sort_copy(I1 first, S1 last, I2 result_first, S2 result_last,
//                           Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                   // Since C++20
//
//     template<input_range R1, random_access_range R2, class Comp = ranges::less,
//              class Proj1 = identity, class Proj2 = identity>
//       requires indirectly_copyable<iterator_t<R1>, iterator_t<R2>> &&
//                sortable<iterator_t<R2>, Comp, Proj2> &&
//                indirect_strict_weak_order<Comp, projected<iterator_t<R1>, Proj1>,
//                                           projected<iterator_t<R2>, Proj2>>
//       constexpr partial_sort_copy_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>>
//         partial_sort_copy(R1&& r, R2&& result_r, Comp comp = {},
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

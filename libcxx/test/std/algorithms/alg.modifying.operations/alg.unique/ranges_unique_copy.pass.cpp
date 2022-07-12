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

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O, class Proj = identity,
//          indirect_equivalence_relation<projected<I, Proj>> C = ranges::equal_to>
//   requires indirectly_copyable<I, O> &&
//            (forward_iterator<I> ||
//             (input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>) ||
//             indirectly_copyable_storable<I, O>)
//   constexpr unique_copy_result<I, O>
//     unique_copy(I first, S last, O result, C comp = {}, Proj proj = {});                         // Since C++20
//
// template<input_range R, weakly_incrementable O, class Proj = identity,
//          indirect_equivalence_relation<projected<iterator_t<R>, Proj>> C = ranges::equal_to>
//   requires indirectly_copyable<iterator_t<R>, O> &&
//            (forward_iterator<iterator_t<R>> ||
//             (input_iterator<O> && same_as<range_value_t<R>, iter_value_t<O>>) ||
//             indirectly_copyable_storable<iterator_t<R>, O>)
//   constexpr unique_copy_result<borrowed_iterator_t<R>, O>
//     unique_copy(R&& r, O result, C comp = {}, Proj proj = {});                                   // Since C++20

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

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

// template<input_iterator I, sentinel_for<I> S, class T, output_iterator<const T&> O,
//          class Proj = identity, indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O>
//   constexpr replace_copy_if_result<I, O>
//     replace_copy_if(I first, S last, O result, Pred pred, const T& new_value,
//                     Proj proj = {});                                                             // Since C++20
//
// template<input_range R, class T, output_iterator<const T&> O, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr replace_copy_if_result<borrowed_iterator_t<R>, O>
//     replace_copy_if(R&& r, O result, Pred pred, const T& new_value,
//                     Proj proj = {});                                                             // Since C++20

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

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

// template<input_iterator I, sentinel_for<I> S,
//          weakly_incrementable O1, weakly_incrementable O2,
//          class Proj = identity, indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O1> && indirectly_copyable<I, O2>
//   constexpr partition_copy_result<I, O1, O2>
//     partition_copy(I first, S last, O1 out_true, O2 out_false, Pred pred,
//                    Proj proj = {});                                                              // Since C++20
//
// template<input_range R, weakly_incrementable O1, weakly_incrementable O2,
//          class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O1> &&
//            indirectly_copyable<iterator_t<R>, O2>
//   constexpr partition_copy_result<borrowed_iterator_t<R>, O1, O2>
//     partition_copy(R&& r, O1 out_true, O2 out_false, Pred pred, Proj proj = {});                 // Since C++20

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

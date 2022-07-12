//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<input_iterator I, sentinel_for<I> S, class T1, class T2,
//          output_iterator<const T2&> O, class Proj = identity>
//   requires indirectly_copyable<I, O> &&
//            indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T1*>
//   constexpr replace_copy_result<I, O>
//     replace_copy(I first, S last, O result, const T1& old_value, const T2& new_value,
//                  Proj proj = {});                                                                // Since C++20
//
// template<input_range R, class T1, class T2, output_iterator<const T2&> O,
//          class Proj = identity>
//   requires indirectly_copyable<iterator_t<R>, O> &&
//            indirect_binary_predicate<ranges::equal_to,
//                                      projected<iterator_t<R>, Proj>, const T1*>
//   constexpr replace_copy_result<borrowed_iterator_t<R>, O>
//     replace_copy(R&& r, O result, const T1& old_value, const T2& new_value,
//                  Proj proj = {});                                                                // Since C++20

// TODO: synopsis

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

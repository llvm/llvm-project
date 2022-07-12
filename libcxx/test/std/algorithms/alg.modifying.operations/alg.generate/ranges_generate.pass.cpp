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

// template<input_or_output_iterator O, sentinel_for<O> S, copy_constructible F>
//   requires invocable<F&> && indirectly_writable<O, invoke_result_t<F&>>
//   constexpr O generate(O first, S last, F gen);                                                  // Since C++20
//
// template<class R, copy_constructible F>
//   requires invocable<F&> && output_range<R, invoke_result_t<F&>>
//   constexpr borrowed_iterator_t<R> generate(R&& r, F gen);                                       // Since C++20

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

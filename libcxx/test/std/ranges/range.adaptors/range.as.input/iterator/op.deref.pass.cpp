//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//   template<input_range V>
//     requires view<V>
//   template<bool Const>
//   class as_input_view<V>::iterator

//    constexpr decltype(auto) operator*() const

#include <cassert>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

constexpr bool test() { return true; }

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

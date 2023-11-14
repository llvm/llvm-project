//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "test.h"
#include "test_iterators.h"
#include <ranges>
#include <type_traits>
#include <utility>

constexpr bool test() {
  using View = InputView<bidirectional_iterator<int*>>;
  static_assert(noexcept(std::declval<std::ranges::stride_view<View>>().stride()));
  static_assert(std::is_same_v<std::ranges::range_difference_t<View>,
                               decltype(std::declval<std::ranges::stride_view<View>>().stride())>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

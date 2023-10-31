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

#include "../test.h"
#include <ranges>

constexpr bool base_noexcept() {
  {
    // If the type of the iterator of the range being strided is default
    // constructible, then the stride view's iterator should be default
    // constructible, too!
    int arr[]                         = {1, 2, 3};
    auto stride                       = std::ranges::stride_view(arr, 1);
    [[maybe_unused]] auto stride_iter = stride.begin();

    static_assert(noexcept(stride_iter.base()));
    static_assert(!noexcept((std::move(stride_iter).base())));
  }

  return true;
}

int main(int, char**) {
  base_noexcept();
  static_assert(base_noexcept());
  return 0;
}

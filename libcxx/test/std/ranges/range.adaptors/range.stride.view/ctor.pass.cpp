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
#include <exception>
#include <ranges>
#include <type_traits>

bool test() {
  // Make sure that a constructor with a negative stride asserts.

  int arr[] = {1, 2, 3};
  ForwardView sc{arr, arr + 3};
  auto sv = std::ranges::stride_view(sc, 0);
  return true;
}

int main(int, char**) {
  test();
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that PSTL algorithms aren't marked [[nodiscard]] when
// _LIBCPP_DISABLE_NODISCARD_EXT is defined

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// UNSUPPORTED: c++03, c++11, c++14

#include <algorithm>
#include <execution>
#include <iterator>

void test() {
  int a[] = {1};
  auto pred = [](auto) { return false; };
  std::all_of(std::execution::par, std::begin(a), std::end(a), pred);
  std::any_of(std::execution::par, std::begin(a), std::end(a), pred);
  std::none_of(std::execution::par, std::begin(a), std::end(a), pred);
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Some basic examples of how zip_tranform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <ranges>

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

int main(int, char**) {
  std::vector v1 = {1, 2};
  std::vector v2 = {4, 5, 6};
  auto ztv       = std::views::zip_transform(std::plus(), v1, v2);
  auto expected  = {5, 7};
  assert(std::ranges::equal(ztv, expected));
  return 0;
}

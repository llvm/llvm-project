//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, std-at-least-c++26

#include <ranges>
#include <cassert>
#include <vector>

constexpr bool test() {
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  auto cv             = std::views::concat(v1, v2);
  static_assert(std::random_access_iterator<decltype(cv.begin())>);
  assert(cv[0] == 1);
  assert(cv[2] == 3);
  assert(cv[3] == 4);
  assert(cv[5] == 6);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

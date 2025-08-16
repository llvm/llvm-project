//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <ranges>
#include <vector>
#include "test_macros.h"

int main(int, char**) {
  std::vector<int> v{1, 2, 3};
  auto r = std::views::counted(std::back_inserter(v), 3);
  auto c = std::views::concat(r);
  // expected-error@*:* {{}}

  return 0;
}

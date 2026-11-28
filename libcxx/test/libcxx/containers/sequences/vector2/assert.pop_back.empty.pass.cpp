//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// pop_back() more than the number of elements in a vector2

// REQUIRES: can-test-hardening-assertions-fast

#include <vector>

#include "check_assertion.h"

int main(int, char**) {
  std::vector2<int> v;
  v.push_back(0);
  v.pop_back();
  TEST_LIBCPP_ASSERT_FAILURE(v.pop_back(), "vector2::pop_back called on an empty vector2");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

TEST_CONSTINIT std::inplace_vector<int, 4> v1;
TEST_CONSTINIT const std::inplace_vector<int, 4> v2;

int main(int, char**) {
  assert(v1.empty());
  assert(v2.empty());

  return 0;
}

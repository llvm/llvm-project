//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, std-at-least-c++26

// <inplace_vector>

// No member named 'regex'.
// XFAIL: LLVM-LIBC-FIXME

#include <inplace_vector>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::inplace_vector<int, 0> c;

  EXPECT_ANY_DEATH((void)c[0]);
  EXPECT_ANY_DEATH((void)c.front());
  EXPECT_ANY_DEATH((void)c.back());
  EXPECT_ANY_DEATH(c.pop_back());
  EXPECT_ANY_DEATH(c.erase(c.begin()));
  EXPECT_ANY_DEATH((void)std::as_const(c)[0]);
  EXPECT_ANY_DEATH((void)std::as_const(c).front());
  EXPECT_ANY_DEATH((void)std::as_const(c).back());

  return 0;
}

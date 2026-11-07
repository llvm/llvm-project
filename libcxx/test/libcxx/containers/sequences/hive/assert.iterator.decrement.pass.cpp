//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive
// REQUIRES: can-test-hardening-assertions-fast

// <hive>

// Decrementing a value-initialized iterator, or begin(): the first slot of the first block.

#include <hive>

#include "check_assertion.h"

int main(int, char**) {
  using H = std::hive<int>;

  H empty;
  TEST_LIBCPP_ASSERT_FAILURE(--empty.end(), "hive: decrementing a non-decrementable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(--empty.cend(), "hive: decrementing a non-decrementable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(H::iterator()--, "hive: decrementing a non-decrementable iterator");

  H h(std::hive_limits(3, 3));
  h.insert(1);
  TEST_LIBCPP_ASSERT_FAILURE(--h.begin(), "hive: decrementing a non-decrementable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(h.begin()--, "hive: decrementing a non-decrementable iterator");

  // Two blocks: the first slot of the first block is still begin().
  for (int n = 0; n != 3; ++n) {
    h.insert(n);
  }
  TEST_LIBCPP_ASSERT_FAILURE(--h.begin(), "hive: decrementing a non-decrementable iterator");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for thrd_sleep.
///
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_sleep.h"

#include "hdr/types/struct_timespec.h"
#include "test/IntegrationTest/test.h"

void test_sleep_zero() {
  struct timespec req = {0, 0};
  struct timespec rem = {123, 456};
  ASSERT_EQ(LIBC_NAMESPACE::thrd_sleep(&req, &rem), 0);
}

void test_sleep_small() {
  struct timespec req = {0, 1000000}; // 1 ms
  ASSERT_EQ(LIBC_NAMESPACE::thrd_sleep(&req, nullptr), 0);
}

void test_sleep_invalid() {
  struct timespec req = {0, -1};
  ASSERT_TRUE(LIBC_NAMESPACE::thrd_sleep(&req, nullptr) < 0);
}

TEST_MAIN() {
  test_sleep_zero();
  test_sleep_small();
  test_sleep_invalid();
  return 0;
}

//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// make check-libc LIBC_TEST_TARGET=call_ctime VERBOSE=1
#include "src/errno/libc_errno.h"
#include "src/time/ctime.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

static inline char *call_ctime(struct time_t *t) {
  return LIBC_NAMESPACE::ctime(t);
}

TEST(LlvmLibcCtime, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime(nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp0) {
  struct time_t t = 0;
  char* result = call_ctime(&t);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp32Int) {
  struct time_t t = 2147483647;
  char* result = call_ctime(&t);
  ASSERT_STREQ("Tue Jan  19 03:14:07 2038\n", result);
}

TEST(LlvmLibcCtime, InvalidArgument) {
  struct time_t t = 2147483648;
  char* result = call_ctime(&t);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

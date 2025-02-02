//===-- Unittests for time ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_func.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTimeTest, SmokeTest) {
  time_t t1;
  time_t t2 = LIBC_NAMESPACE::time(&t1);
  ASSERT_EQ(t1, t2);
  ASSERT_GT(t1, time_t(0));

  time_t t3 = LIBC_NAMESPACE::time(nullptr);
  ASSERT_GE(t3, t1);
}

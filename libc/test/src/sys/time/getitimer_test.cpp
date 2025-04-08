//===-- Unittests for getitimer -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_itimerval.h"
#include "src/sys/time/getitimer.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/time.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysTimeGetitimerTest, SmokeTest) {
  struct itimerval timer;
  timer.it_value.tv_sec = -1;
  timer.it_value.tv_usec = -1;
  timer.it_interval.tv_sec = -1;
  timer.it_interval.tv_usec = -1;

  ASSERT_THAT(LIBC_NAMESPACE::getitimer(0, &timer),
              returns(EQ(0)).with_errno(EQ(0)));

  ASSERT_TRUE(timer.it_value.tv_sec == 0);
  ASSERT_TRUE(timer.it_value.tv_usec == 0);
  ASSERT_TRUE(timer.it_interval.tv_sec == 0);
  ASSERT_TRUE(timer.it_interval.tv_usec == 0);
}

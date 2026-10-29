//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit test for tzset.
///
//===----------------------------------------------------------------------===//

#include "src/time/tz_variables.h"
#include "src/time/tzset.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTzsetTest, StubCallable) {
  LIBC_NAMESPACE::tzset();

  EXPECT_EQ(LIBC_NAMESPACE::tzname[0], nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::tzname[1], nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::timezone, 0L);
  EXPECT_EQ(LIBC_NAMESPACE::daylight, 0);
}

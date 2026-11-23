//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for strptime.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
#include "src/time/strptime.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrptimeTest, StubReturnsNull) {
  struct tm time;
  char *result;

  result = LIBC_NAMESPACE::strptime("13:23:45", "%T", &time);
  EXPECT_EQ(result, nullptr);
}

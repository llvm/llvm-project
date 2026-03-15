//===-- Unittests for gettimeofday ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_timeval.h"
#include "src/time/gettimeofday.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGettimeofday, SmokeTest) {
  timeval tv;
  int ret = LIBC_NAMESPACE::gettimeofday(&tv, nullptr);
  ASSERT_EQ(ret, 0);
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for gettimeofday.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_timeval.h"
#include "src/sys/time/gettimeofday.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGettimeofdayTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGettimeofdayTest, SmokeTest) {
  timeval tv;
  int ret = LIBC_NAMESPACE::gettimeofday(&tv, nullptr);
  ASSERT_EQ(ret, 0);
  ASSERT_ERRNO_SUCCESS();
}

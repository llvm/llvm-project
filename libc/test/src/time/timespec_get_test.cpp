//===-- Unittests for timespec_get ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/properties/architectures.h"
#include "src/time/timespec_get.h"
#include "test/UnitTest/Test.h"

#include <time.h>

TEST(LlvmLibcTimespecGet, Utc) {
#ifndef LIBC_TARGET_ARCH_IS_GPU
  timespec ts;
  int result;
  result = LIBC_NAMESPACE::timespec_get(&ts, TIME_UTC);
  ASSERT_EQ(result, TIME_UTC);
  ASSERT_GT(ts.tv_sec, time_t(0));
#endif
}

TEST(LlvmLibcTimespecGet, Unknown) {
#ifndef LIBC_TARGET_ARCH_IS_GPU
  timespec ts;
  int result;
  result = LIBC_NAMESPACE::timespec_get(&ts, 0);
  ASSERT_EQ(result, 0);
#endif
}

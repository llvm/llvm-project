//===-- Unittests for ctime_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/ctime_r.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

#ifdef LIBC_TARGET_OS_IS_LINUX

#include "src/time/linux/localtime_utils.h"
#include "src/time/linux/timezone.h"
#include <unistd.h>

#endif

using namespace LIBC_NAMESPACE::time_constants;

TEST(LlvmLibcCtimeR, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime_r(nullptr, nullptr);
  ASSERT_STREQ(nullptr, result);

  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  result = LIBC_NAMESPACE::ctime_r(nullptr, buffer);
  ASSERT_STREQ(nullptr, result);

  time_t t;
  result = LIBC_NAMESPACE::ctime_r(&t, nullptr);
  ASSERT_STREQ(nullptr, result);
}

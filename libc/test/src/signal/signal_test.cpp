//===-- Unittests for signal ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/signal/raise.h"
#include "src/signal/signal.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSignal, Invalid) {
  libc_errno = 0;
  __llvm_libc::sighandler_t valid = +[](int) {};
  EXPECT_THAT((void *)__llvm_libc::signal(0, valid),
              Fails(EINVAL, (void *)SIG_ERR));
  EXPECT_THAT((void *)__llvm_libc::signal(65, valid),
              Fails(EINVAL, (void *)SIG_ERR));
}

static int sum;
TEST(LlvmLibcSignal, Basic) {
  // In case test get run multiple times.
  sum = 0;
  ASSERT_NE(__llvm_libc::signal(SIGUSR1, +[](int) { sum++; }),
            SIG_ERR);
  ASSERT_THAT(__llvm_libc::raise(SIGUSR1), Succeeds());
  EXPECT_EQ(sum, 1);
  for (int i = 0; i < 10; i++)
    ASSERT_THAT(__llvm_libc::raise(SIGUSR1), Succeeds());
  EXPECT_EQ(sum, 11);
}

//===-- Interactive unittests for select ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/select/select.h"
#include "src/unistd/read.h"
#include "test/UnitTest/Test.h"

#include <sys/select.h>
#include <unistd.h>

// This test is not be run automatically as part of the libc testsuite.
// Instead, one has to run it manually and press a key on the keyboard
// to make the test succeed.
TEST(LlvmLibcSelectTest, ReadStdinAfterSelect) {
  LIBC_NAMESPACE::libc_errno = 0;
  constexpr int STDIN_FD = 0;
  fd_set set;
  FD_ZERO(&set);
  FD_SET(STDIN_FD, &set);
  struct timeval zero {
    0, 0
  }; // No wait
  struct timeval hr {
    3600, 0
  }; // Wait for an hour.

  // Zero timeout means we don't wait for input. So, select should return
  // immediately.
  int count =
      LIBC_NAMESPACE::select(STDIN_FD + 1, &set, nullptr, nullptr, &zero);
  // The set should indicate that stdin is NOT ready for reading.
  ASSERT_EQ(0, FD_ISSET(STDIN_FD, &set));

  FD_SET(STDIN_FD, &set);
  // Wait for an hour and give the user a chance to hit a key.
  count = LIBC_NAMESPACE::select(STDIN_FD + 1, &set, nullptr, nullptr, &hr);
  ASSERT_EQ(count, 1);
  // The set should indicate that stdin is ready for reading.
  ASSERT_EQ(1, FD_ISSET(STDIN_FD, &set));

  // Verify that atleast one character can be read.
  char c;
  ASSERT_EQ(LIBC_NAMESPACE::read(STDIN_FD, &c, 1), ssize_t(1));
}

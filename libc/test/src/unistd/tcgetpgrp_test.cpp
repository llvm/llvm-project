//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for tcgetpgrp.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/tcgetpgrp.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcTcGetPgrpTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// NOTE: We are not testing the kernel behavior, only how we wire up
// and wrap kernel syscalls in libc interface (on Linux we wrap ioctl)
// and propagate errors. It's hard to get good coverage of tcgetpgrp
// in unit tests, as the test process is not typically associated with
// a terminal. We can achieve this with fork()-ing a child process
// and creating the pseudo-terminal fixture, but this complexity is
// not justified.

TEST_F(LlvmLibcTcGetPgrpTest, BadFd) {
  ASSERT_THAT(LIBC_NAMESPACE::tcgetpgrp(-1), Fails<pid_t>(EBADF));
}

TEST_F(LlvmLibcTcGetPgrpTest, NonTerminalFd) {
  constexpr const char *FILENAME = "tcgetpgrp.test";
  auto test_file = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(test_file, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::tcgetpgrp(fd), Fails<pid_t>(ENOTTY));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(test_file), Succeeds(0));
}

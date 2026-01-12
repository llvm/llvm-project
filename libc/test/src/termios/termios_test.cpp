//===-- Unittests for a bunch of functions in termios.h -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/termios/cfgetispeed.h"
#include "src/termios/cfgetospeed.h"
#include "src/termios/cfsetispeed.h"
#include "src/termios/cfsetospeed.h"
#include "src/termios/tcgetattr.h"
#include "src/termios/tcgetsid.h"
#include "src/termios/tcsetattr.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <termios.h>

using LlvmLibcTermiosTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

// We just list a bunch of smoke tests here as it is not possible to
// test functionality at the least because we want to run the tests
// from ninja/make which change the terminal behavior.

TEST_F(LlvmLibcTermiosTest, SpeedSmokeTest) {
  struct termios t;
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, B50), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetispeed(&t), speed_t(B50));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, B75), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetospeed(&t), speed_t(B75));

  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, ~CBAUD), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, ~CBAUD), Fails(EINVAL));
}

TEST_F(LlvmLibcTermiosTest, GetAttrSmokeTest) {
  struct termios t;
  int fd = LIBC_NAMESPACE::open("/dev/tty", O_RDONLY);
  if (fd < 0) {
    // When /dev/tty is not available, no point continuing
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::tcgetattr(fd, &t), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcTermiosTest, TcGetSidSmokeTest) {
  int fd = LIBC_NAMESPACE::open("/dev/tty", O_RDONLY);
  if (fd < 0) {
    // When /dev/tty is not available, no point continuing
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::tcgetsid(fd),
              returns(GT(pid_t(0))).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

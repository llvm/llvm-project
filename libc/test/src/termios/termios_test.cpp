//===-- Unittests for a bunch of functions in termios.h -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/termios/cfgetispeed.h"
#include "src/termios/cfgetospeed.h"
#include "src/termios/cfsetispeed.h"
#include "src/termios/cfsetospeed.h"
#include "src/termios/tcgetattr.h"
#include "src/termios/tcgetsid.h"
#include "src/termios/tcsetattr.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <termios.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

// We just list a bunch of smoke tests here as it is not possible to
// test functionality at the least because we want to run the tests
// from ninja/make which change the terminal behavior.

TEST(LlvmLibcTermiosTest, SpeedSmokeTest) {
  struct termios t;
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, B50), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetispeed(&t), speed_t(B50));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, B75), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetospeed(&t), speed_t(B75));

  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, ~CBAUD), Fails(EINVAL));
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, ~CBAUD), Fails(EINVAL));
}

TEST(LlvmLibcTermiosTest, GetAttrSmokeTest) {
  struct termios t;
  libc_errno = 0;
  int fd = LIBC_NAMESPACE::open("/dev/tty", O_RDONLY);
  if (fd < 0)
    return; // When /dev/tty is not available, no point continuing.
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(LIBC_NAMESPACE::tcgetattr(fd, &t), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::close(fd), 0);
}

TEST(LlvmLibcTermiosTest, TcGetSidSmokeTest) {
  libc_errno = 0;
  int fd = LIBC_NAMESPACE::open("/dev/tty", O_RDONLY);
  if (fd < 0)
    return; // When /dev/tty is not available, no point continuing.
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(LIBC_NAMESPACE::tcgetsid(fd), pid_t(0));
  ASSERT_EQ(LIBC_NAMESPACE::close(fd), 0);
}

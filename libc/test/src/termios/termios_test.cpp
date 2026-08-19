//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for functions in termios.h.
///
//===----------------------------------------------------------------------===//

#include "hdr/termios_macros.h"
#include "hdr/types/struct_termios.h"
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

using LlvmLibcTermiosTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

// We just list a bunch of smoke tests here as it is not possible to
// test functionality at the least because we want to run the tests
// from ninja/make which change the terminal behavior.

TEST_F(LlvmLibcTermiosTest, SpeedSmokeTest) {
  termios t;
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, B50), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetispeed(&t), speed_t(B50));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, B75), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetospeed(&t), speed_t(B75));

  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, ~CBAUD), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, ~CBAUD), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, 12345), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, 12345), Fails(EINVAL));
#if B50 != 1
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, 4096), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, 4096), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, 1), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, 1), Fails(EINVAL));
#endif
}

TEST_F(LlvmLibcTermiosTest, GetAttrSmokeTest) {
  termios t;
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

TEST_F(LlvmLibcTermiosTest, SplitSpeedTest) {
  int fd = LIBC_NAMESPACE::open("/dev/ptmx", O_RDWR);
  if (fd < 0) {
    // Gracefully skip if /dev/ptmx is not available
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();

  termios t;
  ASSERT_THAT(LIBC_NAMESPACE::tcgetattr(fd, &t), Succeeds(0));

  // 1. Test setting split speeds.
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t, B50), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::cfsetospeed(&t, B75), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::tcsetattr(fd, TCSANOW, &t), Succeeds(0));

  termios t2;
  ASSERT_THAT(LIBC_NAMESPACE::tcgetattr(fd, &t2), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetispeed(&t2), speed_t(B50));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetospeed(&t2), speed_t(B75));

  // 2. Test input speed 0 fallback.
  ASSERT_THAT(LIBC_NAMESPACE::cfsetispeed(&t2, 0), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::tcsetattr(fd, TCSANOW, &t2), Succeeds(0));

  termios t3;
  ASSERT_THAT(LIBC_NAMESPACE::tcgetattr(fd, &t3), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::cfgetospeed(&t3), speed_t(B75));
  // Under POSIX, if input speed was set to 0, it must be the same as output
  // speed.
  ASSERT_EQ(LIBC_NAMESPACE::cfgetispeed(&t3), speed_t(B75));

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

//===-- Unittests for a bunch of functions in termios.h -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/termios/cfgetispeed.h"
#include "src/termios/cfgetospeed.h"
#include "src/termios/cfsetispeed.h"
#include "src/termios/cfsetospeed.h"
#include "src/termios/tcgetattr.h"
#include "src/termios/tcgetsid.h"
#include "src/termios/tcsetattr.h"
#include "src/unistd/close.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <termios.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

// We just list a bunch of smoke tests here as it is not possible to
// test functionality at the least because we want to run the tests
// from ninja/make which change the terminal behavior.

TEST(LlvmLibcTermiosTest, SpeedSmokeTest) {
  struct termios t;
  errno = 0;
  ASSERT_THAT(__llvm_libc::cfsetispeed(&t, B50), Succeeds(0));
  ASSERT_EQ(__llvm_libc::cfgetispeed(&t), speed_t(B50));
  ASSERT_THAT(__llvm_libc::cfsetospeed(&t, B75), Succeeds(0));
  ASSERT_EQ(__llvm_libc::cfgetospeed(&t), speed_t(B75));

  errno = 0;
  ASSERT_THAT(__llvm_libc::cfsetispeed(&t, ~CBAUD), Fails(EINVAL));
  errno = 0;
  ASSERT_THAT(__llvm_libc::cfsetospeed(&t, ~CBAUD), Fails(EINVAL));
}

TEST(LlvmLibcTermiosTest, GetAttrSmokeTest) {
  struct termios t;
  errno = 0;
  int fd = __llvm_libc::open("/dev/tty", O_RDONLY);
  if (fd < 0)
    return; // When /dev/tty is not available, no point continuing.
  ASSERT_EQ(errno, 0);
  ASSERT_THAT(__llvm_libc::tcgetattr(fd, &t), Succeeds(0));
  ASSERT_EQ(__llvm_libc::close(fd), 0);
}

TEST(LlvmLibcTermiosTest, TcGetSidSmokeTest) {
  errno = 0;
  int fd = __llvm_libc::open("/dev/tty", O_RDONLY);
  if (fd < 0)
    return; // When /dev/tty is not available, no point continuing.
  ASSERT_EQ(errno, 0);
  ASSERT_GT(__llvm_libc::tcgetsid(fd), pid_t(0));
  ASSERT_EQ(__llvm_libc::close(fd), 0);
}

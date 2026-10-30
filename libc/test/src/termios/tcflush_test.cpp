//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for tcflush.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/poll_macros.h"
#include "hdr/sys_ioctl_macros.h"
#include "hdr/termios_macros.h"
#include "hdr/types/struct_pollfd.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/poll/poll.h"
#include "src/stdio/snprintf.h"
#include "src/sys/ioctl/ioctl.h"
#include "src/termios/tcflush.h"
#include "src/unistd/close.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

namespace {

bool has_pending_input(int fd) {
  int pending = 0;
  LIBC_NAMESPACE::ioctl(fd, FIONREAD, &pending);
  return pending > 0;
}

} // namespace

// Each test gets a pseudoterminal with a line of input waiting on the
// subsidiary side. The descriptors are closed in TearDown, so no path out of
// a test or of the setup leaks them. When there is no pseudoterminal to use,
// have_pty stays false and the tests return without checking anything.
class LlvmLibcTcflushTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  int manager = -1;
  int subsidiary = -1;
  bool have_pty = false;

  void SetUp() override {
    LIBC_NAMESPACE::testing::ErrnoCheckingTest::SetUp();
    have_pty = open_pseudoterminal_with_input();
    if (have_pty)
      ASSERT_ERRNO_SUCCESS();
    else
      LIBC_NAMESPACE::libc_errno = 0;
  }

  void TearDown() override {
    if (subsidiary >= 0)
      LIBC_NAMESPACE::close(subsidiary);
    if (manager >= 0)
      LIBC_NAMESPACE::close(manager);
    LIBC_NAMESPACE::testing::ErrnoCheckingTest::TearDown();
  }

private:
  bool open_pseudoterminal_with_input() {
    manager = LIBC_NAMESPACE::open("/dev/ptmx", O_RDWR | O_NOCTTY);
    if (manager < 0)
      return false;

    int unlock = 0;
    unsigned int number = 0;
    if (LIBC_NAMESPACE::ioctl(manager, TIOCSPTLCK, &unlock) != 0 ||
        LIBC_NAMESPACE::ioctl(manager, TIOCGPTN, &number) != 0)
      return false;

    char path[32];
    LIBC_NAMESPACE::snprintf(path, sizeof(path), "/dev/pts/%u", number);
    subsidiary = LIBC_NAMESPACE::open(path, O_RDWR | O_NOCTTY);
    if (subsidiary < 0)
      return false;

    if (LIBC_NAMESPACE::write(manager, "input\n", 6) != 6)
      return false;

    // The terminal hands input to the subsidiary side in the background, and a
    // line is readable once the newline has been through.
    struct pollfd pfd = {subsidiary, POLLIN, 0};
    return LIBC_NAMESPACE::poll(&pfd, 1, 5000) == 1;
  }
};

TEST_F(LlvmLibcTcflushTest, OutputFlushKeepsInput) {
  if (!have_pty)
    return;

  ASSERT_TRUE(has_pending_input(subsidiary));
  ASSERT_THAT(LIBC_NAMESPACE::tcflush(subsidiary, TCOFLUSH), Succeeds(0));
  ASSERT_TRUE(has_pending_input(subsidiary));
}

TEST_F(LlvmLibcTcflushTest, InputFlushDropsInput) {
  if (!have_pty)
    return;

  ASSERT_TRUE(has_pending_input(subsidiary));
  ASSERT_THAT(LIBC_NAMESPACE::tcflush(subsidiary, TCIFLUSH), Succeeds(0));
  ASSERT_FALSE(has_pending_input(subsidiary));
}

TEST_F(LlvmLibcTcflushTest, BothFlushDropsInput) {
  if (!have_pty)
    return;

  ASSERT_TRUE(has_pending_input(subsidiary));
  ASSERT_THAT(LIBC_NAMESPACE::tcflush(subsidiary, TCIOFLUSH), Succeeds(0));
  ASSERT_FALSE(has_pending_input(subsidiary));
}

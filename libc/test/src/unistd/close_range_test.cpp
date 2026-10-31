//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for close_range.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/fcntl.h"
#include "src/unistd/close.h"
#include "src/unistd/close_range.h"
#include "src/unistd/pipe.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcCloseRangeTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

namespace {

bool create_adjacent_fds(int fds[2]) {
  if (LIBC_NAMESPACE::pipe(fds) != 0)
    return false;
  if (fds[0] > fds[1]) {
    int tmp = fds[0];
    fds[0] = fds[1];
    fds[1] = tmp;
  }
  while (fds[0] + 1 != fds[1]) {
    int new_fd = LIBC_NAMESPACE::fcntl(fds[0], F_DUPFD, fds[1]);
    if (new_fd < 0) {
      LIBC_NAMESPACE::close(fds[0]);
      LIBC_NAMESPACE::close(fds[1]);
      return false;
    }
    if (LIBC_NAMESPACE::close(fds[0]) != 0) {
      LIBC_NAMESPACE::close(fds[1]);
      LIBC_NAMESPACE::close(new_fd);
      return false;
    }
    fds[0] = fds[1];
    fds[1] = new_fd;
  }
  return true;
}

} // namespace

TEST_F(LlvmLibcCloseRangeTest, InvalidArgs) {
  // first > last should fail with EINVAL.
  ASSERT_THAT(LIBC_NAMESPACE::close_range(10, 5, 0), Fails(EINVAL));

  // Unsupported or invalid flags should fail with EINVAL.
  ASSERT_THAT(LIBC_NAMESPACE::close_range(10, 10, ~0U), Fails(EINVAL));
}

TEST_F(LlvmLibcCloseRangeTest, ClosePipes) {
  int fds[2];
  ASSERT_TRUE(create_adjacent_fds(fds));

  ASSERT_THAT(LIBC_NAMESPACE::close_range(static_cast<unsigned int>(fds[0]),
                                          static_cast<unsigned int>(fds[1]), 0),
              Succeeds(0));

  // Both should now be closed.
  // Using fcntl(F_GETFD) is safer than close() as it does not risk closing an
  // unrelated FD.
  ASSERT_THAT(LIBC_NAMESPACE::fcntl(fds[0], F_GETFD), Fails(EBADF));
  ASSERT_THAT(LIBC_NAMESPACE::fcntl(fds[1], F_GETFD), Fails(EBADF));
}

// CLOSE_RANGE_CLOEXEC was introduced in Linux kernel 5.11.
TEST_F(LlvmLibcCloseRangeTest, CloexecFlag) {
#ifdef CLOSE_RANGE_CLOEXEC
  int fds[2];
  ASSERT_TRUE(create_adjacent_fds(fds));
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fds[0]), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::close(fds[1]), Succeeds(0));
  });

  ASSERT_THAT(LIBC_NAMESPACE::close_range(static_cast<unsigned int>(fds[0]),
                                          static_cast<unsigned int>(fds[1]),
                                          CLOSE_RANGE_CLOEXEC),
              Succeeds(0));

  // Descriptors should remain open and have FD_CLOEXEC set.
  int flags0 = LIBC_NAMESPACE::fcntl(fds[0], F_GETFD);
  ASSERT_GT(flags0, -1);
  EXPECT_NE(flags0 & FD_CLOEXEC, 0);

  int flags1 = LIBC_NAMESPACE::fcntl(fds[1], F_GETFD);
  ASSERT_GT(flags1, -1);
  EXPECT_NE(flags1 & FD_CLOEXEC, 0);
#endif
}

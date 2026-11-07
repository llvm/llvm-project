//===-- Unittests for posix_close -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/unistd_macros.h"
#include "src/fcntl/open.h"
#include "src/unistd/posix_close.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

class LlvmLibcPosixCloseTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  static constexpr const char *FILENAME = "posix_close.test";
  const char *test_path = nullptr;

  void SetUp() override { test_path = libc_make_test_file_path(FILENAME); }

  void TearDown() override {
    if (test_path)
      LIBC_NAMESPACE::unlink(test_path);
  }
};

TEST_F(LlvmLibcPosixCloseTest, InvalidFlag) {
  // Invalid flag should fail with EINVAL.
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, ~0), Fails<int>(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, 0x1234), Fails<int>(EINVAL));
}

TEST_F(LlvmLibcPosixCloseTest, InvalidFd) {
  // Valid flag (0) with invalid fd should fail with EBADF.
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, 0), Fails<int>(EBADF));
#ifdef POSIX_CLOSE_RESTART
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, POSIX_CLOSE_RESTART),
              Fails<int>(EBADF));
#endif
}

TEST_F(LlvmLibcPosixCloseTest, CloseSuccess) {
  int fd = LIBC_NAMESPACE::open(test_path, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  EXPECT_THAT(LIBC_NAMESPACE::posix_close(fd, 0), Succeeds(0));

  // Subsequent operation on closed fd should fail with EBADF.
  char buf[1];
  EXPECT_THAT(LIBC_NAMESPACE::read(fd, buf, 1), Fails<ssize_t>(EBADF));
}

TEST_F(LlvmLibcPosixCloseTest, CloseWithRestartFlag) {
#ifdef POSIX_CLOSE_RESTART
  int fd = LIBC_NAMESPACE::open(test_path, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  EXPECT_THAT(LIBC_NAMESPACE::posix_close(fd, POSIX_CLOSE_RESTART),
              Succeeds(0));

  // Subsequent operation on closed fd should fail with EBADF.
  char buf[1];
  EXPECT_THAT(LIBC_NAMESPACE::read(fd, buf, 1), Fails<ssize_t>(EBADF));
#endif
}

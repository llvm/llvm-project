//===-- Unittests for getrlimit and setrlimit -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/resource/getrlimit.h"
#include "src/sys/resource/setrlimit.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/resource.h>
#include <sys/stat.h>

TEST(LlvmLibcResourceLimitsTest, SetNoFileLimit) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  // The test strategy is to first create initialize two file descriptors
  // successfully. Next, close the files and set the file descriptor limit
  // to 4. This will allow us to open one of those file but not the other.

  constexpr const char *TEST_FILE1 = "testdata/resource_limits1.test";
  constexpr const char *TEST_FILE2 = "testdata/resource_limits2.test";
  LIBC_NAMESPACE::libc_errno = 0;

  int fd1 = LIBC_NAMESPACE::open(TEST_FILE1, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd1, 0);
  ASSERT_ERRNO_SUCCESS();
  int fd2 = LIBC_NAMESPACE::open(TEST_FILE2, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd2, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::close(fd1), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd2), Succeeds(0));

  struct rlimit limits {
    4, 4
  };
  ASSERT_THAT(LIBC_NAMESPACE::setrlimit(RLIMIT_NOFILE, &limits), Succeeds(0));

  // One can now only open one of the files successfully.
  fd1 = LIBC_NAMESPACE::open(TEST_FILE1, O_RDONLY);
  ASSERT_GT(fd1, 0);
  ASSERT_ERRNO_SUCCESS();
  fd2 = LIBC_NAMESPACE::open(TEST_FILE2, O_RDONLY);
  ASSERT_LT(fd2, 0);
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::close(fd1), Succeeds(0));

  fd2 = LIBC_NAMESPACE::open(TEST_FILE2, O_RDONLY);
  ASSERT_GT(fd2, 0);
  ASSERT_ERRNO_SUCCESS();
  fd1 = LIBC_NAMESPACE::open(TEST_FILE1, O_RDONLY);
  ASSERT_LT(fd1, 0);
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::close(fd2), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE1), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE2), Succeeds(0));

  struct rlimit current_limits;
  ASSERT_THAT(LIBC_NAMESPACE::getrlimit(RLIMIT_NOFILE, &current_limits),
              Succeeds(0));
  ASSERT_EQ(current_limits.rlim_cur, rlim_t(4));
  ASSERT_EQ(current_limits.rlim_max, rlim_t(4));
}

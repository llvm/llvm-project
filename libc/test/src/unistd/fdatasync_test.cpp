//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fdatasync.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fdatasync.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFdatasyncTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFdatasyncTest, BasicSync) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "fdatasync.test";
  auto test_file = libc_make_test_file_path(FILENAME);
  constexpr const char WRITE_DATA[] = "hello, fdatasync";
  constexpr ssize_t WRITE_SIZE = sizeof(WRITE_DATA);

  int fd = LIBC_NAMESPACE::open(test_file, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  ASSERT_THAT(LIBC_NAMESPACE::write(fd, WRITE_DATA, WRITE_SIZE),
              Succeeds(WRITE_SIZE));
  ASSERT_THAT(LIBC_NAMESPACE::fdatasync(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(test_file), Succeeds(0));
}

TEST_F(LlvmLibcFdatasyncTest, BadFd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::fdatasync(-1), Fails(EBADF));
}

//===-- Unittests for fstat -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/stat/fstat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

TEST(LlvmLibcFStatTest, CreatAndReadMode) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

  // The test file is initially writable. We open it for writing and ensure
  // that it indeed can be opened for writing. Next, we close the file and
  // make it readonly using chmod. We test that chmod actually succeeded by
  // trying to open the file for writing and failing.
  constexpr const char *TEST_FILE = "testdata/fstat.test";
  errno = 0;

  int fd = __llvm_libc::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(errno, 0);

  struct stat statbuf;
  ASSERT_THAT(__llvm_libc::fstat(fd, &statbuf), Succeeds(0));

  ASSERT_EQ(int(statbuf.st_mode), int(S_IRWXU | S_IFREG));

  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcFStatTest, NonExistentFile) {
  errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  struct stat statbuf;
  ASSERT_THAT(__llvm_libc::fstat(-1, &statbuf), Fails(EBADF));
  errno = 0;
}

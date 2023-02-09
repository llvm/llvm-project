//===-- Unittests for fchmodat --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/stat/fchmodat.h"
#include "src/unistd/close.h"
#include "src/unistd/write.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

TEST(LlvmLibcFchmodatTest, ChangeAndOpen) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

  // The test file is initially writable. We open it for writing and ensure
  // that it indeed can be opened for writing. Next, we close the file and
  // make it readonly using chmod. We test that chmod actually succeeded by
  // trying to open the file for writing and failing.
  constexpr const char *TEST_FILE = "testdata/fchmodat.test";
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE_BASENAME = "fchmodat.test";
  const char WRITE_DATA[] = "fchmodat test";
  constexpr ssize_t WRITE_SIZE = ssize_t(sizeof(WRITE_DATA));
  errno = 0;

  int fd = __llvm_libc::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(errno, 0);
  ASSERT_EQ(__llvm_libc::write(fd, WRITE_DATA, sizeof(WRITE_DATA)), WRITE_SIZE);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  int dirfd = __llvm_libc::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dirfd, 0);
  ASSERT_EQ(errno, 0);

  EXPECT_THAT(__llvm_libc::fchmodat(dirfd, TEST_FILE_BASENAME, S_IRUSR, 0),
              Succeeds(0));

  // Opening for writing should fail.
  EXPECT_EQ(__llvm_libc::open(TEST_FILE, O_APPEND | O_WRONLY), -1);
  EXPECT_NE(errno, 0);
  errno = 0;
  // But opening for reading should succeed.
  fd = __llvm_libc::open(TEST_FILE, O_APPEND | O_RDONLY);
  EXPECT_GT(fd, 0);
  EXPECT_EQ(errno, 0);

  EXPECT_THAT(__llvm_libc::close(fd), Succeeds(0));
  EXPECT_THAT(__llvm_libc::fchmodat(dirfd, TEST_FILE_BASENAME, S_IRWXU, 0),
              Succeeds(0));

  EXPECT_THAT(__llvm_libc::close(dirfd), Succeeds(0));
}

TEST(LlvmLibcFchmodatTest, NonExistentFile) {
  errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::fchmodat(AT_FDCWD, "non-existent-file", S_IRUSR, 0),
              Fails(ENOENT));
  errno = 0;
}

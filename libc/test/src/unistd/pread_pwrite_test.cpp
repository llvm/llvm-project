//===-- Unittests for pread and pwrite ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fsync.h"
#include "src/unistd/pread.h"
#include "src/unistd/pwrite.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>

TEST(LlvmLibcUniStd, PWriteAndPReadBackTest) {
  // The strategy here is that we first create a file and write to it. Next,
  // we open that file again and write at an offset. Finally, we open the
  // file again and pread at an offset and make sure that only expected data
  // is being read back. This also confirms that pwrite happened successfully.
  constexpr const char HELLO[] = "hello";
  constexpr int HELLO_SIZE = sizeof(HELLO);
  constexpr off_t OFFSET = 3;
  constexpr const char OFFSET_TEXT[] = "helhello";
  constexpr int OFFSET_TEXT_SIZE = sizeof(OFFSET_TEXT);

  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *TEST_FILE = "testdata/pread_pwrite.test";
  int fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::write(fd, HELLO, HELLO_SIZE), Succeeds(HELLO_SIZE));
  ASSERT_THAT(__llvm_libc::fsync(fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_WRONLY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::pwrite(fd, HELLO, HELLO_SIZE, OFFSET),
              Succeeds(HELLO_SIZE));
  ASSERT_THAT(__llvm_libc::fsync(fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  char read_buf[OFFSET_TEXT_SIZE];
  ASSERT_THAT(__llvm_libc::pread(fd, read_buf, HELLO_SIZE, OFFSET),
              Succeeds(HELLO_SIZE));
  EXPECT_STREQ(read_buf, HELLO);
  ASSERT_THAT(__llvm_libc::pread(fd, read_buf, OFFSET_TEXT_SIZE, 0),
              Succeeds(OFFSET_TEXT_SIZE));
  EXPECT_STREQ(read_buf, OFFSET_TEXT);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcUniStd, PWriteFails) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(__llvm_libc::pwrite(-1, "", 1, 0), Fails(EBADF));
}

TEST(LlvmLibcUniStd, PReadFails) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(__llvm_libc::pread(-1, nullptr, 1, 0), Fails(EBADF));
}

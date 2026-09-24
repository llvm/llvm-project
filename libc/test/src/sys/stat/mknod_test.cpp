//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for mknod.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/stat/mknod.h"
#include "src/sys/stat/stat.h"
#include "src/sys/stat/umask.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMknodTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMknodTest, CreateAndRemoveRegularFile) {
  auto TEST_FILE = libc_make_test_file_path("mknod_reg.test");
  constexpr mode_t FILE_MODE = S_IRUSR | S_IWUSR;

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mknod(TEST_FILE, S_IFREG | FILE_MODE, 0),
              Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISREG(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FILE_MODE));
}

TEST_F(LlvmLibcMknodTest, CreateAndRemoveFifo) {
  auto TEST_FIFO = libc_make_test_file_path("mknod_fifo.test");
  constexpr mode_t FIFO_MODE = S_IRUSR | S_IWUSR;

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mknod(TEST_FIFO, S_IFIFO | FIFO_MODE, 0),
              Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FIFO), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FIFO, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISFIFO(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FIFO_MODE));
}

TEST_F(LlvmLibcMknodTest, NonExistentPath) {
  auto BAD_PATH = libc_make_test_file_path("non-existent-dir/mknod.test");
  ASSERT_THAT(LIBC_NAMESPACE::mknod(BAD_PATH, S_IFREG | 0644, 0),
              Fails(ENOENT));
}

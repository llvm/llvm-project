//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for mkfifo.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/sys/stat/mkfifo.h"
#include "src/sys/stat/stat.h"
#include "src/sys/stat/umask.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMkfifoTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkfifoTest, CreateAndRemove) {
  constexpr const char *FILENAME = "testdata/mkfifo.testfifo";
  auto TEST_FIFO = libc_make_test_file_path(FILENAME);
  constexpr mode_t FIFO_MODE = S_IRUSR | S_IWUSR;

  // Clear the file creation mask so that the resulting permission bits are
  // exactly the requested ones.
  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mkfifo(TEST_FIFO, FIFO_MODE), Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  // The created file must be a FIFO with the requested permissions, which
  // verifies that S_IFIFO was requested and that mode was passed along.
  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FIFO, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISFIFO(statbuf.st_mode));
  ASSERT_EQ(static_cast<int>(statbuf.st_mode & 07777),
            static_cast<int>(FIFO_MODE));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FIFO), Succeeds(0));
}

TEST_F(LlvmLibcMkfifoTest, BadPath) {
  constexpr const char *FILENAME = "testdata/non-existent-dir/mkfifo.testfifo";
  auto TEST_FIFO = libc_make_test_file_path(FILENAME);

  ASSERT_THAT(LIBC_NAMESPACE::mkfifo(TEST_FIFO, S_IRUSR | S_IWUSR),
              Fails(ENOENT));
}

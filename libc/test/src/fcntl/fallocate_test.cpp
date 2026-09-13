//===-- Unittests for fallocate -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/fallocate.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/fstat.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcFallocateTest, BasicAllocate) {
  constexpr char TEST_FILE[] = "testdata/fallocate_basic.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY | O_TRUNC, 0600);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup_fd([&] {
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  });

  // Mode 0: expand file size to offset + len (4096 bytes)
  ASSERT_THAT(LIBC_NAMESPACE::fallocate(fd, 0, 0, 4096), Succeeds(0));

  struct stat st;
  ASSERT_EQ(LIBC_NAMESPACE::fstat(fd, &st), 0);
  ASSERT_EQ(static_cast<off_t>(st.st_size), static_cast<off_t>(4096));
}

TEST(LlvmLibcFallocateTest, KeepSize) {
  constexpr char TEST_FILE[] = "testdata/fallocate_keep_size.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY | O_TRUNC, 0600);
  ASSERT_GT(fd, 0);

  // Pre-allocate space at offset 8192, length 4096, but keep file size at 0
  ASSERT_THAT(LIBC_NAMESPACE::fallocate(fd, FALLOC_FL_KEEP_SIZE, 8192, 4096),
              Succeeds(0));

  struct stat st;
  ASSERT_EQ(LIBC_NAMESPACE::fstat(fd, &st), 0);
  ASSERT_EQ(static_cast<off_t>(st.st_size), static_cast<off_t>(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

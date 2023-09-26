//===-- Unittests for link ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/link.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcLinkTest, CreateAndUnlink) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/link.test";
  constexpr const char *TEST_FILE_LINK = "testdata/link.test.link";

  // The test strategy is as follows:
  //   1. Create a normal file
  //   2. Create a link to that file.
  //   3. Open the link to check that the link was created.
  //   4. Cleanup the file and its link.
  libc_errno = 0;
  int write_fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::link(TEST_FILE, TEST_FILE_LINK), Succeeds(0));

  int link_fd = LIBC_NAMESPACE::open(TEST_FILE_LINK, O_PATH);
  ASSERT_GT(link_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(link_fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_LINK), Succeeds(0));
}

TEST(LlvmLibcLinkTest, LinkToNonExistentFile) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(
      LIBC_NAMESPACE::link("testdata/non-existent-file", "testdata/bad-link"),
      Fails(ENOENT));
}

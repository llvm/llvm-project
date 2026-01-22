//===-- Unittests for chown -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/unistd/chown.h"
#include "src/unistd/close.h"
#include "src/unistd/getgid.h"
#include "src/unistd/getuid.h"
#include "src/unistd/unlink.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"
#include <sys/stat.h>

using LlvmLibcChownTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcChownTest, ChownSuccess) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  uid_t my_uid = LIBC_NAMESPACE::getuid();
  gid_t my_gid = LIBC_NAMESPACE::getgid();
  constexpr const char *FILENAME = "chown.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  // Create a test file.
  int write_fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);

  // Change the ownership of the file.
  ASSERT_THAT(LIBC_NAMESPACE::chown(TEST_FILE, my_uid, my_gid), Succeeds(0));

  // Close the file descriptor.
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  // Clean up the test file.
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST_F(LlvmLibcChownTest, ChownNonExistentFile) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::chown("non-existent-file", 1000, 1000),
              Fails(ENOENT));
}

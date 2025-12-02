//===-- Unittests for fchown ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fchown.h"
#include "src/unistd/getgid.h"
#include "src/unistd/getuid.h"
#include "src/unistd/unlink.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"
#include <sys/stat.h>

using LlvmLibcFchownTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFchownTest, FchownSuccess) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  uid_t my_uid = LIBC_NAMESPACE::getuid();
  gid_t my_gid = LIBC_NAMESPACE::getgid();
  constexpr const char *FILENAME = "fchown.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  // Create a test file.
  int write_fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);

  // Change the ownership of the file.
  ASSERT_THAT(LIBC_NAMESPACE::fchown(write_fd, my_uid, my_gid), Succeeds(0));

  // Close the file descriptor.
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  // Clean up the test file.
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST_F(LlvmLibcFchownTest, FchownInvalidFileDescriptor) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::fchown(-1, 1000, 1000), Fails(EBADF));
}

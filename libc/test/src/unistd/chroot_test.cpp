//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for chroot
///
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdir.h"
#include "src/unistd/chroot.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/sys_stat_macros.h"

using LlvmLibcChrootTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcChrootTest, NonExistentDir) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  constexpr const char *FILENAME = "non_existent_dir_chroot.test";
  auto TEST_DIR = libc_make_test_file_path(FILENAME);
  ASSERT_THAT(LIBC_NAMESPACE::chroot(TEST_DIR), Fails(ENOENT));
}

TEST_F(LlvmLibcChrootTest, ChangeRoot) {
  constexpr const char *FILENAME = "chroot.testdir";
  auto TEST_DIR = libc_make_test_file_path(FILENAME);
  LIBC_NAMESPACE::rmdir(TEST_DIR);
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(TEST_DIR, S_IRWXU),
              LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds(0));
  // try to chroot to this folder. Fails without permissions, but that's about
  // all we can reasonably test.
  int res = LIBC_NAMESPACE::chroot(TEST_DIR);
  if (res == 0) {
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_ERRNO_EQ(EPERM);
  }
  LIBC_NAMESPACE::rmdir(TEST_DIR);
  libc_errno = 0;
}

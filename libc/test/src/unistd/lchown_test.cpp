//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for lchown.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/getgid.h"
#include "src/unistd/getuid.h"
#include "src/unistd/lchown.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcLchownTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcLchownTest, LchownSuccessRegularFile) {
  uid_t my_uid = LIBC_NAMESPACE::getuid();
  gid_t my_gid = LIBC_NAMESPACE::getgid();
  constexpr const char *FILENAME = "lchown_regular.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  int write_fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });

  // Change ownership to current user/group on a regular file.
  ASSERT_THAT(LIBC_NAMESPACE::lchown(TEST_FILE, my_uid, my_gid), Succeeds(0));

  // Passing -1 for owner and group leaves them unchanged without error.
  ASSERT_THAT(LIBC_NAMESPACE::lchown(TEST_FILE, static_cast<uid_t>(-1),
                                     static_cast<gid_t>(-1)),
              Succeeds(0));
}

TEST_F(LlvmLibcLchownTest, LchownSuccessSymlink) {
  uid_t my_uid = LIBC_NAMESPACE::getuid();
  gid_t my_gid = LIBC_NAMESPACE::getgid();
  constexpr const char *TARGET_FILENAME = "lchown_target.test";
  constexpr const char *LINK_FILENAME = "lchown_symlink.test";
  auto TARGET_FILE = libc_make_test_file_path(TARGET_FILENAME);
  auto LINK_FILE = libc_make_test_file_path(LINK_FILENAME);

  int write_fd = LIBC_NAMESPACE::open(TARGET_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_target(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TARGET_FILE), Succeeds(0)); });

  // Create a symbolic link to the target file.
  ASSERT_THAT(LIBC_NAMESPACE::symlink(TARGET_FILE, LINK_FILE), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_link(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(LINK_FILE), Succeeds(0)); });

  // lchown should operate on the symbolic link itself.
  ASSERT_THAT(LIBC_NAMESPACE::lchown(LINK_FILE, my_uid, my_gid), Succeeds(0));

  // Passing -1 for owner and group leaves them unchanged without error.
  ASSERT_THAT(LIBC_NAMESPACE::lchown(LINK_FILE, static_cast<uid_t>(-1),
                                     static_cast<gid_t>(-1)),
              Succeeds(0));
}

TEST_F(LlvmLibcLchownTest, LchownNonExistentFile) {
  auto BAD_PATH = libc_make_test_file_path("non_existent_file_lchown.test");
  ASSERT_THAT(LIBC_NAMESPACE::lchown(BAD_PATH, 1000, 1000), Fails(ENOENT));
}

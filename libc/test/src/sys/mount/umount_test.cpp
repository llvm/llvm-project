//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for umount.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/mount/mount.h"
#include "src/sys/mount/umount.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcUmountTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// mount and umount need CAP_SYS_ADMIN, so unprivileged runs see EPERM,
// container LSMs (e.g. AppArmor/SELinux) report EACCES, and seccomp filters may
// report ENOSYS.

TEST_F(LlvmLibcUmountTest, UmountMountedDir) {
  auto TEST_DIR = libc_make_test_file_path("umount.testdir");
  LIBC_NAMESPACE::rmdir(TEST_DIR);
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(TEST_DIR, S_IRWXU), Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    LIBC_NAMESPACE::rmdir(TEST_DIR);
    libc_errno = 0;
  });

  int ret = LIBC_NAMESPACE::mount("none", TEST_DIR, "tmpfs", 0, nullptr);
  if (ret == 0) {
    ASSERT_ERRNO_SUCCESS();
    ASSERT_THAT(LIBC_NAMESPACE::umount(TEST_DIR), Succeeds(0));
  } else {
    ASSERT_THAT(ret, Fails(any_of(EPERM, EACCES, ENOSYS)));
  }
}

TEST_F(LlvmLibcUmountTest, UmountNotAMountPoint) {
  auto TEST_DIR = libc_make_test_file_path("umount_not_mounted.testdir");
  LIBC_NAMESPACE::rmdir(TEST_DIR);
  libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(TEST_DIR, S_IRWXU), Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    LIBC_NAMESPACE::rmdir(TEST_DIR);
    libc_errno = 0;
  });

  ASSERT_THAT(LIBC_NAMESPACE::umount(TEST_DIR),
              Fails(any_of(EINVAL, EPERM, EACCES, ENOSYS)));
}

//===-- Unittests for umask -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/fstat.h"
#include "src/sys/stat/umask.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcUmaskTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// umask cannot fail (POSIX: "shall always be successful"), so there is no
// error-path test here.

TEST_F(LlvmLibcUmaskTest, SetAndGetPrevious) {
  // Each call must return the mask installed by the previous one. Three
  // distinct values are used so that no assertion degenerates into a tautology
  // if the inherited umask happens to equal one of them.
  mode_t original = LIBC_NAMESPACE::umask(S_IRWXG | S_IRWXO);
  EXPECT_EQ(LIBC_NAMESPACE::umask(S_IWGRP | S_IWOTH),
            static_cast<mode_t>(S_IRWXG | S_IRWXO));
  EXPECT_EQ(LIBC_NAMESPACE::umask(S_IRWXO),
            static_cast<mode_t>(S_IWGRP | S_IWOTH));

  // umask must not touch errno.
  libc_errno = EDOM;
  LIBC_NAMESPACE::umask(original); // also restores the inherited mask
  ASSERT_ERRNO_EQ(EDOM);           // ASSERT_ERRNO_EQ resets errno afterwards
}

TEST_F(LlvmLibcUmaskTest, AffectsCreatedFilePermissions) {
  // Both cases are required: a single mask can coincide with the process's
  // inherited umask, letting an implementation that ignores its argument pass.
  constexpr mode_t ALL_PERMS = S_IRWXU | S_IRWXG | S_IRWXO;

  { // umask 0022 -> 0755
    auto TEST_FILE = libc_make_test_file_path("testdata/umask_0022.test");
    LIBC_NAMESPACE::unlink(TEST_FILE); // best-effort: drop a stale file
    libc_errno = 0;

    mode_t old_mask = LIBC_NAMESPACE::umask(S_IWGRP | S_IWOTH);
    int fd =
        LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_EXCL | O_WRONLY, ALL_PERMS);
    LIBC_NAMESPACE::umask(old_mask); // restore right after the file is created
    ASSERT_GE(fd, 0);
    ASSERT_ERRNO_SUCCESS();

    struct stat statbuf;
    ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &statbuf), Succeeds(0));
    EXPECT_EQ(
        statbuf.st_mode & ALL_PERMS,
        static_cast<mode_t>(S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH));
    EXPECT_EQ(statbuf.st_mode & static_cast<mode_t>(S_IFMT),
              static_cast<mode_t>(S_IFREG));

    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  }
  { // umask 0077 -> 0700
    auto TEST_FILE = libc_make_test_file_path("testdata/umask_0077.test");
    LIBC_NAMESPACE::unlink(TEST_FILE); // best-effort: drop a stale file
    libc_errno = 0;

    mode_t old_mask = LIBC_NAMESPACE::umask(S_IRWXG | S_IRWXO);
    int fd =
        LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_EXCL | O_WRONLY, ALL_PERMS);
    LIBC_NAMESPACE::umask(old_mask); // restore right after the file is created
    ASSERT_GE(fd, 0);
    ASSERT_ERRNO_SUCCESS();

    struct stat statbuf;
    ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &statbuf), Succeeds(0));
    EXPECT_EQ(statbuf.st_mode & ALL_PERMS, static_cast<mode_t>(S_IRWXU));
    EXPECT_EQ(statbuf.st_mode & static_cast<mode_t>(S_IFMT),
              static_cast<mode_t>(S_IFREG));

    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  }
}

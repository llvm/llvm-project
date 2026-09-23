//===-- Unittests for utimensat -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/fcntl/open.h"
#include "src/stdio/remove.h"
#include "src/sys/stat/stat.h"
#include "src/sys/stat/utimensat.h"
#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcUtimensatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// SUCCESS: Takes a file and successfully updates
// its last access and modified times.
TEST_F(LlvmLibcUtimensatTest, ChangeTimesSpecific) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILE_PATH = "utimensat_pass.test";
  auto TEST_FILE = libc_make_test_file_path(FILE_PATH);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // make a dummy timespec struct
  struct timespec times[2];
  times[0].tv_sec = 54321;
  times[0].tv_nsec = 12345000;
  times[1].tv_sec = 43210;
  times[1].tv_nsec = 23456000;

  // ensure utimensat succeeds
  ASSERT_THAT(LIBC_NAMESPACE::utimensat(AT_FDCWD, TEST_FILE, times, 0),
              Succeeds(0));

  // verify the times values against stat of the TEST_FILE
  struct stat statbuf;
  ASSERT_EQ(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), 0);

  // seconds
  ASSERT_EQ(statbuf.st_atim.tv_sec, times[0].tv_sec);
  ASSERT_EQ(statbuf.st_mtim.tv_sec, times[1].tv_sec);

  // nanoseconds
  ASSERT_EQ(statbuf.st_atim.tv_nsec, times[0].tv_nsec);
  ASSERT_EQ(statbuf.st_mtim.tv_nsec, times[1].tv_nsec);

  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
}

// FAILURE: Invalid values in the timespec struct
// to check that utimensat rejects it.
TEST_F(LlvmLibcUtimensatTest, InvalidNanoseconds) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILE_PATH = "utimensat_fail.test";
  auto TEST_FILE = libc_make_test_file_path(FILE_PATH);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // make a dummy timespec struct
  // populated with bad nsec values
  struct timespec times[2];
  times[0].tv_sec = 54321;
  times[0].tv_nsec = 4567;
  times[1].tv_sec = 43210;
  times[1].tv_nsec = 1000000000; // invalid

  // ensure utimensat fails
  ASSERT_THAT(LIBC_NAMESPACE::utimensat(AT_FDCWD, TEST_FILE, times, 0),
              Fails(EINVAL));

  // check for failure on
  // the other possible bad values
  times[0].tv_sec = 54321;
  times[0].tv_nsec = -4567; // invalid
  times[1].tv_sec = 43210;
  times[1].tv_nsec = 1000;

  // ensure utimensat fails once more
  ASSERT_THAT(LIBC_NAMESPACE::utimensat(AT_FDCWD, TEST_FILE, times, 0),
              Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
}

// SUCCESS: Test UTIME_NOW and UTIME_OMIT macros
TEST_F(LlvmLibcUtimensatTest, UtimeNowAndOmit) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILE_PATH = "utimensat_now.test";
  auto TEST_FILE = libc_make_test_file_path(FILE_PATH);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  struct timespec times[2];
  times[0].tv_sec = 0;
  times[0].tv_nsec = UTIME_NOW;
  times[1].tv_sec = 0;
  times[1].tv_nsec = UTIME_OMIT;

  ASSERT_THAT(LIBC_NAMESPACE::utimensat(AT_FDCWD, TEST_FILE, times, 0),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
}

//===-- Unittests for utimes ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/struct_timeval.h"
#include "src/fcntl/open.h"
#include "src/stdio/remove.h"
#include "src/sys/stat/stat.h"
#include "src/sys/time/utimes.h"
#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcUtimesTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// SUCCESS: Takes a file and successfully updates
// its last access and modified times.
TEST_F(LlvmLibcUtimesTest, ChangeTimesSpecific) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILE_PATH = "utimes_pass.test";
  auto TEST_FILE = libc_make_test_file_path(FILE_PATH);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // make a dummy timeval struct
  struct timeval times[2];
  times[0].tv_sec = 54321;
  times[0].tv_usec = 12345;
  times[1].tv_sec = 43210;
  times[1].tv_usec = 23456;

  // ensure utimes succeeds
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Succeeds(0));

  // verify the times values against stat of the TEST_FILE
  struct stat statbuf;
  ASSERT_EQ(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), 0);

  // seconds
  ASSERT_EQ(statbuf.st_atim.tv_sec, times[0].tv_sec);
  ASSERT_EQ(statbuf.st_mtim.tv_sec, times[1].tv_sec);

  // microseconds
  ASSERT_EQ(statbuf.st_atim.tv_nsec,
            static_cast<long>(times[0].tv_usec * 1000));
  ASSERT_EQ(statbuf.st_mtim.tv_nsec,
            static_cast<long>(times[1].tv_usec * 1000));

  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
}

// FAILURE: Invalid values in the timeval struct
// to check that utimes rejects it.
TEST_F(LlvmLibcUtimesTest, InvalidMicroseconds) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILE_PATH = "utimes_fail.test";
  auto TEST_FILE = libc_make_test_file_path(FILE_PATH);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // make a dummy timeval struct
  // populated with bad usec values
  struct timeval times[2];
  times[0].tv_sec = 54321;
  times[0].tv_usec = 4567;
  times[1].tv_sec = 43210;
  times[1].tv_usec = 1000000; // invalid

  // ensure utimes fails
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Fails(EINVAL));

  // check for failure on
  // the other possible bad values

  times[0].tv_sec = 54321;
  times[0].tv_usec = -4567; // invalid
  times[1].tv_sec = 43210;
  times[1].tv_usec = 1000;

  // ensure utimes fails once more
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
}

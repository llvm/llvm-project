//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for timerfd_gettime and timerfd_settime.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_timerfd_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_itimerspec.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/time_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/time/clock_gettime.h"
#include "src/sys/timerfd/timerfd_create.h"
#include "src/sys/timerfd/timerfd_gettime.h"
#include "src/sys/timerfd/timerfd_settime.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;
using LlvmLibcTimerfdGettimeTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using NsecT = decltype(timespec::tv_nsec);

TEST_F(LlvmLibcTimerfdGettimeTest, SetAndGetRelative) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::timerfd_create(CLOCK_MONOTONIC,
                                                  TFD_CLOEXEC | TFD_NONBLOCK),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  // Newly created timer should be disarmed.
  struct itimerspec curr_val = {{1, 1}, {1, 1}};
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_gettime(fd, &curr_val), Succeeds(0));
  EXPECT_EQ(curr_val.it_value.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_value.tv_nsec, static_cast<NsecT>(0));
  EXPECT_EQ(curr_val.it_interval.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_interval.tv_nsec, static_cast<NsecT>(0));

  // Arm the timer with a relative timeout and periodic interval.
  struct itimerspec new_val = {};
  new_val.it_value.tv_sec = 60;
  new_val.it_value.tv_nsec = 0;
  new_val.it_interval.tv_sec = 2;
  new_val.it_interval.tv_nsec = 500000000;

  struct itimerspec old_val = {{1, 1}, {1, 1}};
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, 0, &new_val, &old_val),
              Succeeds(0));
  EXPECT_EQ(old_val.it_value.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(old_val.it_value.tv_nsec, static_cast<NsecT>(0));
  EXPECT_EQ(old_val.it_interval.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(old_val.it_interval.tv_nsec, static_cast<NsecT>(0));

  ASSERT_THAT(LIBC_NAMESPACE::timerfd_gettime(fd, &curr_val), Succeeds(0));
  EXPECT_TRUE(curr_val.it_value.tv_sec > 0 || curr_val.it_value.tv_nsec > 0);
  EXPECT_LE(curr_val.it_value.tv_sec, static_cast<time_t>(60));
  EXPECT_EQ(curr_val.it_interval.tv_sec, static_cast<time_t>(2));
  EXPECT_EQ(curr_val.it_interval.tv_nsec, static_cast<NsecT>(500000000));

  // Disarm the timer and verify old_value returns the previous setting.
  struct itimerspec disarm_val = {};
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, 0, &disarm_val, &old_val),
              Succeeds(0));
  EXPECT_TRUE(old_val.it_value.tv_sec > 0 || old_val.it_value.tv_nsec > 0);
  EXPECT_EQ(old_val.it_interval.tv_sec, static_cast<time_t>(2));
  EXPECT_EQ(old_val.it_interval.tv_nsec, static_cast<NsecT>(500000000));

  ASSERT_THAT(LIBC_NAMESPACE::timerfd_gettime(fd, &curr_val), Succeeds(0));
  EXPECT_EQ(curr_val.it_value.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_value.tv_nsec, static_cast<NsecT>(0));
  EXPECT_EQ(curr_val.it_interval.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_interval.tv_nsec, static_cast<NsecT>(0));
}

TEST_F(LlvmLibcTimerfdGettimeTest, SetAndGetAbsolute) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  struct timespec now = {};
  auto clock_res =
      LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &now);
  ASSERT_TRUE(clock_res.has_value());

  struct itimerspec new_val = {};
  new_val.it_value.tv_sec = now.tv_sec + 60;
  new_val.it_value.tv_nsec = now.tv_nsec;
  ASSERT_THAT(
      LIBC_NAMESPACE::timerfd_settime(fd, TFD_TIMER_ABSTIME, &new_val, nullptr),
      Succeeds(0));

  struct itimerspec curr_val = {};
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_gettime(fd, &curr_val), Succeeds(0));
  EXPECT_TRUE(curr_val.it_value.tv_sec > 0 || curr_val.it_value.tv_nsec > 0);
  EXPECT_LE(curr_val.it_value.tv_sec, static_cast<time_t>(60));
  EXPECT_EQ(curr_val.it_interval.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_interval.tv_nsec, static_cast<NsecT>(0));

  // Set an absolute one-shot timer in the past so it expires immediately.
  struct itimerspec past_val = {};
  past_val.it_value.tv_sec = 0;
  past_val.it_value.tv_nsec = 1;
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, TFD_TIMER_ABSTIME, &past_val,
                                              nullptr),
              Succeeds(0));

  uint64_t expirations = 0;
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, &expirations, sizeof(expirations)),
              Succeeds<ssize_t>(sizeof(expirations)));
  EXPECT_EQ(expirations, static_cast<uint64_t>(1));

  // One-shot timer should be disarmed after expiring.
  ASSERT_THAT(LIBC_NAMESPACE::timerfd_gettime(fd, &curr_val), Succeeds(0));
  EXPECT_EQ(curr_val.it_value.tv_sec, static_cast<time_t>(0));
  EXPECT_EQ(curr_val.it_value.tv_nsec, static_cast<NsecT>(0));
}

TEST_F(LlvmLibcTimerfdGettimeTest, InvalidArgs) {
  struct itimerspec val = {};
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_gettime(-1, &val), Fails(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_settime(-1, 0, &val, nullptr),
              Fails(EBADF));

  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  EXPECT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, -1, &val, nullptr),
              Fails(EINVAL));

  struct itimerspec bad_nsec = {};
  bad_nsec.it_value.tv_sec = 1;
  bad_nsec.it_value.tv_nsec = 1000000000;
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, 0, &bad_nsec, nullptr),
              Fails(EINVAL));

  bad_nsec.it_value.tv_nsec = -1;
  EXPECT_THAT(LIBC_NAMESPACE::timerfd_settime(fd, 0, &bad_nsec, nullptr),
              Fails(EINVAL));
}

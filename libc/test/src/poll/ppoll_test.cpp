//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for ppoll.
///
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "hdr/poll_macros.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_pollfd.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/scope.h"
#include "src/poll/ppoll.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPPollTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPPollTest, SmokeTest) {
  timespec ts{0, 0};
  int ret = LIBC_NAMESPACE::ppoll(nullptr, 0, &ts, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
}

TEST_F(LlvmLibcPPollTest, SmokeFailureTest) {
  int ret = LIBC_NAMESPACE::ppoll(nullptr, UINT_MAX, nullptr, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(-1, ret);
}

TEST_F(LlvmLibcPPollTest, TimeoutNotMutated) {
  const timespec orig_ts{0, 0};
  timespec ts = orig_ts;
  int ret = LIBC_NAMESPACE::ppoll(nullptr, 0, &ts, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
  ASSERT_EQ(ts.tv_sec, orig_ts.tv_sec);
  ASSERT_EQ(ts.tv_nsec, orig_ts.tv_nsec);
}

TEST_F(LlvmLibcPPollTest, WithSigmask) {
  timespec ts{0, 0};
  sigset_t mask{};
  int ret = LIBC_NAMESPACE::ppoll(nullptr, 0, &ts, &mask);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
}

TEST_F(LlvmLibcPPollTest, PipeReadiness) {
  int pipefd[2];
  ASSERT_EQ(LIBC_NAMESPACE::pipe(pipefd), 0);
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    LIBC_NAMESPACE::close(pipefd[0]);
    LIBC_NAMESPACE::close(pipefd[1]);
  });

  pollfd pfd{pipefd[0], POLLIN, 0};
  timespec ts{0, 0};
  int ret = LIBC_NAMESPACE::ppoll(&pfd, 1, &ts, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
  ASSERT_EQ(0, static_cast<int>(pfd.revents));

  char c = 'x';
  ASSERT_EQ(LIBC_NAMESPACE::write(pipefd[1], &c, 1), static_cast<ssize_t>(1));
  ASSERT_ERRNO_SUCCESS();

  ret = LIBC_NAMESPACE::ppoll(&pfd, 1, &ts, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(1, ret);
  ASSERT_EQ(POLLIN, pfd.revents & POLLIN);

  char buf = 0;
  ASSERT_EQ(LIBC_NAMESPACE::read(pipefd[0], &buf, 1), static_cast<ssize_t>(1));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ('x', buf);
}

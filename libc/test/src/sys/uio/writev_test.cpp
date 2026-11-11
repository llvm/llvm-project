//===-- Unittests for writev ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_iovec.h"
#include "src/sys/uio/writev.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysUioWritevTest, SmokeTest) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds());

  const char *data = "Hello, World!\n";
  struct iovec iov[2];
  iov[0].iov_base = const_cast<char *>(data);
  iov[0].iov_len = 7;
  iov[1].iov_base = const_cast<char *>(data + 7);
  iov[1].iov_len = 8;
  ASSERT_THAT(LIBC_NAMESPACE::writev(pipefd[1], iov, 2),
              returns(EQ(ssize_t(15))).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds());

  char buf[16];
  ASSERT_THAT(LIBC_NAMESPACE::read(pipefd[0], buf, 15),
              returns(EQ(ssize_t(15))).with_errno(EQ(0)));
  buf[15] = '\0';
  EXPECT_STREQ(buf, data);
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds());
}

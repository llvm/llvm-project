//===-- Unittests for readv -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_iovec.h"
#include "src/sys/uio/readv.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysUioReadvTest, SmokeTest) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds());

  const char data[] = "Hello, World!\n";
  ASSERT_THAT(LIBC_NAMESPACE::write(pipefd[1], data, sizeof(data)),
              returns(EQ(sizeof(data))).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds());

  char buf0[2];
  char buf1[3];
  struct iovec iov[2];
  iov[0].iov_base = buf0;
  iov[0].iov_len = 1;
  iov[1].iov_base = buf1;
  iov[1].iov_len = 2;
  ASSERT_THAT(LIBC_NAMESPACE::readv(pipefd[0], iov, 2),
              returns(EQ(ssize_t(3))).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds());
}

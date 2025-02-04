//===-- Unittests for writev ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/uio/writev.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysUioWritevTest, SmokeTest) {
  int fd = LIBC_NAMESPACE::open("/dev/null", O_WRONLY);
  ASSERT_THAT(fd, returns(GT(0)).with_errno(EQ(0)));
  const char *data = "Hello, World!\n";
  struct iovec iov[2];
  iov[0].iov_base = const_cast<char *>(data);
  iov[0].iov_len = 7;
  iov[1].iov_base = const_cast<char *>(data + 7);
  iov[1].iov_len = 8;
  ASSERT_THAT(LIBC_NAMESPACE::writev(fd, iov, 2),
              returns(EQ(15)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}

//===-- Unittests for pipe2
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe2.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcPipe2Test, ReadAndWriteViaPipe2) {
  int pipefd[2];
  int flags;

  LIBC_NAMESPACE::libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  // Create pipe(2) with flags set to 0
  flags = 0;
  ASSERT_NE(LIBC_NAMESPACE::pipe2(pipefd, flags), -1);
  ASSERT_ERRNO_SUCCESS();

  // Write something via the pipe and read from other end
  constexpr char MESSAGE[] = "Hello from the write end!";
  constexpr size_t MESSAGE_SIZE = sizeof(MESSAGE);
  char buf[MESSAGE_SIZE];
  ASSERT_EQ(ssize_t(MESSAGE_SIZE),
            LIBC_NAMESPACE::write(pipefd[1], MESSAGE, MESSAGE_SIZE));
  ASSERT_EQ(ssize_t(MESSAGE_SIZE),
            LIBC_NAMESPACE::read(pipefd[0], buf, MESSAGE_SIZE));
  ASSERT_STREQ(buf, MESSAGE);

  // Close the pipe file descriptors
  ASSERT_NE(LIBC_NAMESPACE::close(pipefd[0]), -1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(LIBC_NAMESPACE::close(pipefd[1]), -1);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcPipe2Test, Pipe2InvalidFlags) {
  int invalidflags = O_CREAT | O_PATH | O_SYNC;
  int pipefd[2];

  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::pipe2(pipefd, invalidflags), Fails(EINVAL));
}

TEST(LlvmLibcPipe2Test, Pipe2InvalidPipeFD) {
  int flags = 0;
  int invalidpipefd[1];

  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::pipe2(invalidpipefd, flags), Fails(EFAULT));
}

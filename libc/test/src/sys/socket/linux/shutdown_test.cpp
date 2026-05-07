//===-- Unittests for shutdown --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/ssize_t.h"
#include "src/sys/socket/shutdown.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcShutdownTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcShutdownTest, ShutWrProducesEOF) {
  int sv[2];
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sv),
              Succeeds(0));

  // Shut down write on sv[0].
  ASSERT_THAT(LIBC_NAMESPACE::shutdown(sv[0], SHUT_WR), Succeeds(0));

  // Reading from sv[1] should report end-of-file by returning 0.
  char read_buf[10];
  ASSERT_THAT(LIBC_NAMESPACE::read(sv[1], read_buf, sizeof(read_buf)),
              Succeeds<ssize_t>(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(sv[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sv[1]), Succeeds(0));
}

TEST_F(LlvmLibcShutdownTest, ShutRdPreventsReading) {
  int sv[2];
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sv),
              Succeeds(0));

  // Shut down read on sv[0].
  ASSERT_THAT(LIBC_NAMESPACE::shutdown(sv[0], SHUT_RD), Succeeds(0));

  // Reading from sv[0] should report end-of-file by returning 0.
  char read_buf[10];
  ASSERT_THAT(LIBC_NAMESPACE::read(sv[0], read_buf, sizeof(read_buf)),
              Succeeds<ssize_t>(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(sv[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sv[1]), Succeeds(0));
}

TEST_F(LlvmLibcShutdownTest, ShutRdWrDoesBoth) {
  int sv[2];
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sv),
              Succeeds(0));

  // Shut down read and write on sv[0].
  ASSERT_THAT(LIBC_NAMESPACE::shutdown(sv[0], SHUT_RDWR), Succeeds(0));

  // Both descriptors should report end-of-file by returning 0.
  char read_buf[10];
  ASSERT_THAT(LIBC_NAMESPACE::read(sv[0], read_buf, sizeof(read_buf)),
              Succeeds<ssize_t>(0));
  ASSERT_THAT(LIBC_NAMESPACE::read(sv[1], read_buf, sizeof(read_buf)),
              Succeeds<ssize_t>(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(sv[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sv[1]), Succeeds(0));
}

TEST_F(LlvmLibcShutdownTest, FailsOnInvalidSocket) {
  ASSERT_THAT(LIBC_NAMESPACE::shutdown(-1, SHUT_WR), Fails(EBADF));
}

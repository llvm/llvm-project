//===-- Unittests for process_vm_readv ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_iovec.h"
#include "src/sys/uio/process_vm_readv.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysUioProcessVmReadvTest, SmokeTest) {
  pid_t pid = LIBC_NAMESPACE::getpid();
  char src[] = "Hello, World!\n";

  struct iovec remote_iov[1];
  remote_iov[0].iov_base = src;
  remote_iov[0].iov_len = sizeof(src);

  char buf0[7];
  char buf1[8];
  struct iovec local_iov[2];
  local_iov[0].iov_base = buf0;
  local_iov[0].iov_len = 7;
  local_iov[1].iov_base = buf1;
  local_iov[1].iov_len = sizeof(src) - 7;

  ASSERT_THAT(
      LIBC_NAMESPACE::process_vm_readv(pid, local_iov, 2, remote_iov, 1, 0),
      returns(EQ(ssize_t(sizeof(src)))).with_errno(EQ(0)));

  EXPECT_EQ(buf0[0], 'H');
  EXPECT_EQ(buf0[6], ' ');
  EXPECT_EQ(buf1[0], 'W');
}

TEST(LlvmLibcSysUioProcessVmReadvTest, InvalidFlags) {
  pid_t pid = LIBC_NAMESPACE::getpid();
  char buf[8];
  struct iovec iov = {buf, sizeof(buf)};
  ASSERT_THAT(LIBC_NAMESPACE::process_vm_readv(pid, &iov, 1, &iov, 1, 1),
              returns(EQ(ssize_t(-1))).with_errno(EQ(EINVAL)));
}
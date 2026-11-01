//===-- Unittests for process_vm_writev -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_iovec.h"
#include "src/sys/uio/process_vm_writev.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysUioProcessVmWritevTest, SmokeTest) {
  pid_t pid = LIBC_NAMESPACE::getpid();
  char dest[15] = {};

  struct iovec remote_iov[1];
  remote_iov[0].iov_base = dest;
  remote_iov[0].iov_len = sizeof(dest);

  char part0[] = "Hello, ";
  char part1[] = "World!\n";
  struct iovec local_iov[2];
  local_iov[0].iov_base = part0;
  local_iov[0].iov_len = 7;
  local_iov[1].iov_base = part1;
  local_iov[1].iov_len = 8; // includes '\n' and '\0'

  ASSERT_THAT(
      LIBC_NAMESPACE::process_vm_writev(pid, local_iov, 2, remote_iov, 1, 0),
      returns(EQ(ssize_t(15))).with_errno(EQ(0)));

  EXPECT_EQ(dest[0], 'H');
  EXPECT_EQ(dest[7], 'W');
}

TEST(LlvmLibcSysUioProcessVmWritevTest, InvalidFlags) {
  pid_t pid = LIBC_NAMESPACE::getpid();
  char buf[8];
  struct iovec iov = {buf, sizeof(buf)};
  ASSERT_THAT(LIBC_NAMESPACE::process_vm_writev(pid, &iov, 1, &iov, 1, 1),
              returns(EQ(ssize_t(-1))).with_errno(EQ(EINVAL)));
}
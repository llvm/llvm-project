//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for memfd_create.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_mman_macros.h"
#include "include/llvm-libc-macros/file-seek-macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/fcntl.h"
#include "src/sys/mman/memfd_create.h"
#include "src/unistd/close.h"
#include "src/unistd/lseek.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMemfdCreateTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMemfdCreateTest, Basic) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::memfd_create("test_memfd", MFD_CLOEXEC),
              returns(GE(0)).with_errno(EQ(0)));
  LIBC_NAMESPACE::cpp::scope_exit close_fd(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  int flag;
  ASSERT_THAT(flag = LIBC_NAMESPACE::fcntl(fd, F_GETFD),
              returns(GE(0)).with_errno(EQ(0)));
  EXPECT_NE(flag & FD_CLOEXEC, 0);

  ASSERT_THAT(LIBC_NAMESPACE::lseek(fd, 0, SEEK_END), Succeeds(off_t(0)));
}

TEST_F(LlvmLibcMemfdCreateTest, ErrorHandling) {
  // Passing invalid flags should cause EINVAL
  ASSERT_THAT(LIBC_NAMESPACE::memfd_create("test_memfd", 0x80000000),
              Fails(EINVAL));
}

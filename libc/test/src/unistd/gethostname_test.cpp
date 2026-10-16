//===-- Unittests for gethostname -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/gethostname.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetHostNameTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST(LlvmLibcGetHostNameTest, GetCurrHostName) {
  char hostbuffer[1024];
  int ret = LIBC_NAMESPACE::gethostname(hostbuffer, sizeof(hostbuffer));
  ASSERT_NE(ret, -1);
  ASSERT_ERRNO_SUCCESS();

  ret = LIBC_NAMESPACE::gethostname(hostbuffer, 0);
  ASSERT_EQ(ret, -1);
  ASSERT_ERRNO_EQ(ENAMETOOLONG);

  // test for invalid pointer
  char *nptr = nullptr;
  ret = LIBC_NAMESPACE::gethostname(nptr, 1);
  ASSERT_EQ(ret, -1);
  ASSERT_ERRNO_EQ(EFAULT);
}

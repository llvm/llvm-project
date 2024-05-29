//===-- Unittests for prctl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/prctl/prctl.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include <errno.h>
#include <sys/prctl.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSysPrctlTest, GetSetName) {
  char name[17];
  unsigned long name_addr = 0;
  ASSERT_THAT(LIBC_NAMESPACE::prctl(PR_GET_NAME, name_addr, 0, 0, 0),
              Fails(EFAULT, -1));

  name_addr = reinterpret_cast<unsigned long>("libc-test");
  ASSERT_THAT(LIBC_NAMESPACE::prctl(PR_SET_NAME, name_addr, 0, 0, 0),
              Succeeds());

  name_addr = reinterpret_cast<unsigned long>(name);
  ASSERT_THAT(LIBC_NAMESPACE::prctl(PR_GET_NAME, name_addr, 0, 0, 0),
              Succeeds());
  ASSERT_STREQ(name, "libc-test");
}

TEST(LlvmLibcSysPrctlTest, GetTHPDisable) {
  // Manually check errno since the return value logic here is not
  // covered in ErrnoSetterMatcher.
  LIBC_NAMESPACE::libc_errno = 0;
  int ret = LIBC_NAMESPACE::prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_ERRNO_SUCCESS();
  // PR_GET_THP_DISABLE return (as the function result) the current
  // setting of the "THP disable" flag for the calling thread, which
  // is either 1, if the flag is set; or 0, if it is not.
  ASSERT_TRUE(ret == 0 || ret == 1);
}

//===-- Unittests for errno -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcErrnoTest, Basic) {
  int test_val = 123;
  libc_errno = test_val;
#if LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_UNDEFINED &&                            \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SYSTEM_INLINE
  EXPECT_EQ(&libc_errno, LIBC_NAMESPACE::__llvm_libc_errno());
  EXPECT_EQ(*LIBC_NAMESPACE::__llvm_libc_errno(), test_val);
#endif
  ASSERT_ERRNO_EQ(test_val);
}

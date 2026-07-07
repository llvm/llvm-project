//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for if_nametoindex.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/net/if_nametoindex.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcIfNameToIndexTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcIfNameToIndexTest, Loopback) {
  unsigned int idx = LIBC_NAMESPACE::if_nametoindex("lo");
  ASSERT_GT(idx, 0u);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcIfNameToIndexTest, InvalidName) {
  unsigned int idx = LIBC_NAMESPACE::if_nametoindex("nonexistent_if");
  ASSERT_EQ(idx, 0u);
  ASSERT_ERRNO_EQ(ENODEV);
}

TEST_F(LlvmLibcIfNameToIndexTest, EmptyName) {
  unsigned int idx = LIBC_NAMESPACE::if_nametoindex("");
  ASSERT_EQ(idx, 0u);
  ASSERT_ERRNO_EQ(ENODEV);
}

TEST_F(LlvmLibcIfNameToIndexTest, NullPtrDeath) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::if_nametoindex(nullptr); },
               WITH_SIGNAL(-1));
}

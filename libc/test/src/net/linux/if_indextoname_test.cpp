//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for if_indextoname.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "src/net/if_indextoname.h"
#include "src/net/if_nametoindex.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcIfIndexToNameTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcIfIndexToNameTest, Loopback) {
  unsigned int lo_idx = LIBC_NAMESPACE::if_nametoindex("lo");
  ASSERT_GT(lo_idx, 0u);

  char buf[IF_NAMESIZE];
  char *res = LIBC_NAMESPACE::if_indextoname(lo_idx, buf);
  ASSERT_EQ(res, buf);
  ASSERT_STREQ(buf, "lo");
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcIfIndexToNameTest, InvalidIndex) {
  char buf[IF_NAMESIZE];
  char *res = LIBC_NAMESPACE::if_indextoname(-1u, buf);
  ASSERT_EQ(res, static_cast<char *>(nullptr));
  ASSERT_ERRNO_EQ(ENXIO);
}

TEST_F(LlvmLibcIfIndexToNameTest, ZeroIndex) {
  char buf[IF_NAMESIZE];
  char *res = LIBC_NAMESPACE::if_indextoname(0u, buf);
  ASSERT_EQ(res, static_cast<char *>(nullptr));
  ASSERT_ERRNO_EQ(ENXIO);
}

TEST_F(LlvmLibcIfIndexToNameTest, NullPtrDeath) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::if_indextoname(1u, nullptr); },
               WITH_SIGNAL(-1));
}

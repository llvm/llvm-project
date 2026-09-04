//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fnmatch.
///
//===----------------------------------------------------------------------===//

#include "hdr/fnmatch_macros.h"
#include "src/fnmatch/fnmatch.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcFnmatchTest, StubAlwaysReturnsNoMatch) {
  EXPECT_EQ(LIBC_NAMESPACE::fnmatch("test", "test", 0), FNM_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::fnmatch("*", "anything", 0), FNM_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::fnmatch("a", "a", FNM_PATHNAME | FNM_PERIOD),
            FNM_NOMATCH);
}

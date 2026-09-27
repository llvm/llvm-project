//===-- Unittests for iswblank --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswblank.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswblank, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswblank(' '), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswblank('\t'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswblank('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswblank('\n'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswblank('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswblank('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswblank('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswblank(-1), 0);
}

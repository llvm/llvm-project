//===-- Unittests for iswgraph --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswgraph.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswgraph, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswgraph('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswgraph('0'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswgraph('?'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswgraph('!'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswgraph('~'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswgraph(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswgraph('\t'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswgraph('\n'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswgraph('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswgraph(-1), 0);
}

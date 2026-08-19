//===-- Unittests for iswxdigit -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswxdigit.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibciswxdigit, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'1'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'2'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'0'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'B'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswxdigit(L'F'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswxdigit(L'g'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswxdigit(L'h'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswxdigit(L'é'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswxdigit(L' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswxdigit(L'!'), 0);
}

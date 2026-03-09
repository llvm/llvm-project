//===-- Unittests for iswlower --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswlower.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibciswlower, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswlower('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswlower('b'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswlower('z'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswlower('A'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswlower('B'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswlower('Z'), 0);
}

//===-- Unittests for iswupper --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswupper.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswupper, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswupper('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswupper('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper(-1), 0);
}

// Tests for the extended range are located in
// libc/test/src/__support/wctype_utils_test.cpp.

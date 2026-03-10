//===-- Unittests for iswdigit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/wctype/iswdigit.h"

#include "test/UnitTest/Test.h"

// Simple tests, already properly tested in
// libc/test/src/__support/wctype_utils_test.cpp
TEST(LlvmLibciswdigit, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswdigit(L'1'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswdigit(L'2'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswdigit(L'0'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswdigit(L'a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswdigit(L'é'), 0);
}

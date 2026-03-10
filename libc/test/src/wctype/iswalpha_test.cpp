//===-- Unittests for iswalpha --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/wctype_utils.h"
#include "src/wctype/iswalpha.h"

#include "test/UnitTest/Test.h"

// Simple tests, already properly tested in
// libc/test/src/__support/wctype_utils_test.cpp and
// libc/test/src/__support/wctype/wctype_classification_utils_test.cpp
TEST(LlvmLibciswalpha, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswalpha('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswalpha('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha(-1), 0);

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
  EXPECT_NE(LIBC_NAMESPACE::iswalpha(L'á'), 0);
#else
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha(L'á'), 0);
#endif
}

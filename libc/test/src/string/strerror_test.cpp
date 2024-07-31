//===-- Unittests for strerror --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/StringUtil/platform_errors.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/string/strerror.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrErrorTest, KnownErrors) {
  ASSERT_STREQ(LIBC_NAMESPACE::strerror(0), "Success");

  for (auto [i, msg] : LIBC_NAMESPACE::PLATFORM_ERRORS)
    EXPECT_STREQ(LIBC_NAMESPACE::strerror(static_cast<int>(i)), msg.begin());
}

TEST(LlvmLibcStrErrorTest, UnknownErrors) {
  ASSERT_STREQ(LIBC_NAMESPACE::strerror(-1), "Unknown error -1");
  ASSERT_STREQ(LIBC_NAMESPACE::strerror(134), "Unknown error 134");
  ASSERT_STREQ(LIBC_NAMESPACE::strerror(2147483647),
               "Unknown error 2147483647");
  ASSERT_STREQ(LIBC_NAMESPACE::strerror(-2147483648),
               "Unknown error -2147483648");
}

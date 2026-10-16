//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for libintl stub functions.
///
//===----------------------------------------------------------------------===//

#include "src/libintl/textdomain.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcLibintlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcLibintlTest, Textdomain) {
  EXPECT_STREQ(LIBC_NAMESPACE::textdomain(nullptr), "messages");
  EXPECT_STREQ(LIBC_NAMESPACE::textdomain(""), "messages");

  constexpr const char *DOMAIN_NAME = "my_app";
  EXPECT_THAT(LIBC_NAMESPACE::textdomain(DOMAIN_NAME),
              Succeeds(const_cast<char *>(DOMAIN_NAME)));
}

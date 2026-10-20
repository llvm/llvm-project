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

#include "src/libintl/bind_textdomain_codeset.h"
#include "src/libintl/bindtextdomain.h"
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

TEST_F(LlvmLibcLibintlTest, Bindtextdomain) {
  EXPECT_THAT(LIBC_NAMESPACE::bindtextdomain(nullptr, "/custom/dir"),
              Succeeds<char *>(nullptr));
  EXPECT_THAT(LIBC_NAMESPACE::bindtextdomain("", "/custom/dir"),
              Succeeds<char *>(nullptr));
  EXPECT_STREQ(LIBC_NAMESPACE::bindtextdomain("my_app", nullptr),
               "/usr/share/locale");

  constexpr const char *DIR_NAME = "/custom/dir";
  EXPECT_THAT(LIBC_NAMESPACE::bindtextdomain("my_app", DIR_NAME),
              Succeeds(const_cast<char *>(DIR_NAME)));
}

TEST_F(LlvmLibcLibintlTest, BindTextdomainCodeset) {
  EXPECT_THAT(LIBC_NAMESPACE::bind_textdomain_codeset(nullptr, "UTF-8"),
              Succeeds<char *>(nullptr));
  EXPECT_THAT(LIBC_NAMESPACE::bind_textdomain_codeset("", "UTF-8"),
              Succeeds<char *>(nullptr));
  EXPECT_THAT(LIBC_NAMESPACE::bind_textdomain_codeset("my_app", nullptr),
              Succeeds<char *>(nullptr));

  constexpr const char *CODESET = "UTF-8";
  EXPECT_THAT(LIBC_NAMESPACE::bind_textdomain_codeset("my_app", CODESET),
              Succeeds(const_cast<char *>(CODESET)));
}

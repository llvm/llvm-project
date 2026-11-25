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

#include "hdr/locale_macros.h"
#include "hdr/types/locale_t.h"
#include "src/libintl/bind_textdomain_codeset.h"
#include "src/libintl/bindtextdomain.h"
#include "src/libintl/dcgettext.h"
#include "src/libintl/dcgettext_l.h"
#include "src/libintl/dcngettext.h"
#include "src/libintl/dcngettext_l.h"
#include "src/libintl/dgettext.h"
#include "src/libintl/dgettext_l.h"
#include "src/libintl/dngettext.h"
#include "src/libintl/dngettext_l.h"
#include "src/libintl/gettext.h"
#include "src/libintl/gettext_l.h"
#include "src/libintl/ngettext.h"
#include "src/libintl/ngettext_l.h"
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

TEST_F(LlvmLibcLibintlTest, Gettext) {
  constexpr const char *MSG = "hello world";
  locale_t loc = nullptr;

  EXPECT_THAT(LIBC_NAMESPACE::gettext(MSG), Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::gettext_l(MSG, loc),
              Succeeds(const_cast<char *>(MSG)));
}

TEST_F(LlvmLibcLibintlTest, Dgettext) {
  constexpr const char *MSG = "hello world";
  locale_t loc = nullptr;

  EXPECT_THAT(LIBC_NAMESPACE::dgettext("my_app", MSG),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dgettext(nullptr, MSG),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dgettext_l("my_app", MSG, loc),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dgettext_l(nullptr, MSG, loc),
              Succeeds(const_cast<char *>(MSG)));
}

TEST_F(LlvmLibcLibintlTest, Dcgettext) {
  constexpr const char *MSG = "hello world";
  locale_t loc = nullptr;

  EXPECT_THAT(LIBC_NAMESPACE::dcgettext("my_app", MSG, LC_MESSAGES),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dcgettext(nullptr, MSG, LC_MESSAGES),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dcgettext_l("my_app", MSG, LC_MESSAGES, loc),
              Succeeds(const_cast<char *>(MSG)));
  EXPECT_THAT(LIBC_NAMESPACE::dcgettext_l(nullptr, MSG, LC_MESSAGES, loc),
              Succeeds(const_cast<char *>(MSG)));
}

TEST_F(LlvmLibcLibintlTest, Ngettext) {
  constexpr const char *SINGULAR = "one file";
  constexpr const char *PLURAL = "many files";
  locale_t loc = nullptr;

  EXPECT_THAT(LIBC_NAMESPACE::ngettext(SINGULAR, PLURAL, 0),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::ngettext(SINGULAR, PLURAL, 1),
              Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(LIBC_NAMESPACE::ngettext(SINGULAR, PLURAL, 2),
              Succeeds(const_cast<char *>(PLURAL)));

  EXPECT_THAT(LIBC_NAMESPACE::ngettext_l(SINGULAR, PLURAL, 0, loc),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::ngettext_l(SINGULAR, PLURAL, 1, loc),
              Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(LIBC_NAMESPACE::ngettext_l(SINGULAR, PLURAL, 2, loc),
              Succeeds(const_cast<char *>(PLURAL)));
}

TEST_F(LlvmLibcLibintlTest, Dngettext) {
  constexpr const char *SINGULAR = "one file";
  constexpr const char *PLURAL = "many files";
  locale_t loc = nullptr;

  EXPECT_THAT(LIBC_NAMESPACE::dngettext("my_app", SINGULAR, PLURAL, 0),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext("my_app", SINGULAR, PLURAL, 1),
              Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext("my_app", SINGULAR, PLURAL, 2),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext(nullptr, SINGULAR, PLURAL, 1),
              Succeeds(const_cast<char *>(SINGULAR)));

  EXPECT_THAT(LIBC_NAMESPACE::dngettext_l("my_app", SINGULAR, PLURAL, 0, loc),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext_l("my_app", SINGULAR, PLURAL, 1, loc),
              Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext_l("my_app", SINGULAR, PLURAL, 2, loc),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::dngettext_l(nullptr, SINGULAR, PLURAL, 1, loc),
              Succeeds(const_cast<char *>(SINGULAR)));
}

TEST_F(LlvmLibcLibintlTest, Dcngettext) {
  constexpr const char *SINGULAR = "one file";
  constexpr const char *PLURAL = "many files";
  locale_t loc = nullptr;

  EXPECT_THAT(
      LIBC_NAMESPACE::dcngettext("my_app", SINGULAR, PLURAL, 0, LC_MESSAGES),
      Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(
      LIBC_NAMESPACE::dcngettext("my_app", SINGULAR, PLURAL, 1, LC_MESSAGES),
      Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(
      LIBC_NAMESPACE::dcngettext("my_app", SINGULAR, PLURAL, 2, LC_MESSAGES),
      Succeeds(const_cast<char *>(PLURAL)));

  EXPECT_THAT(LIBC_NAMESPACE::dcngettext_l("my_app", SINGULAR, PLURAL, 0,
                                           LC_MESSAGES, loc),
              Succeeds(const_cast<char *>(PLURAL)));
  EXPECT_THAT(LIBC_NAMESPACE::dcngettext_l("my_app", SINGULAR, PLURAL, 1,
                                           LC_MESSAGES, loc),
              Succeeds(const_cast<char *>(SINGULAR)));
  EXPECT_THAT(LIBC_NAMESPACE::dcngettext_l("my_app", SINGULAR, PLURAL, 2,
                                           LC_MESSAGES, loc),
              Succeeds(const_cast<char *>(PLURAL)));
}

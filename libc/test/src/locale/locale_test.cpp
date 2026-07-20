//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for locale functions.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/locale_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/locale/freelocale.h"
#include "src/locale/newlocale.h"
#include "src/locale/uselocale.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLocale = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcLocale, DefaultLocale) {
  locale_t new_locale = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "C", nullptr);
  ASSERT_NE(new_locale, nullptr);
  LIBC_NAMESPACE::cpp::scope_exit free_new(
      [&] { LIBC_NAMESPACE::freelocale(new_locale); });

  locale_t old_locale = LIBC_NAMESPACE::uselocale(new_locale);
  ASSERT_NE(old_locale, nullptr);
  EXPECT_NE(LIBC_NAMESPACE::uselocale(nullptr), nullptr);

  locale_t restored_locale = LIBC_NAMESPACE::uselocale(old_locale);
  EXPECT_NE(restored_locale, nullptr);
}

TEST_F(LlvmLibcLocale, NewLocaleValidation) {
  locale_t loc =
      LIBC_NAMESPACE::newlocale(LC_CTYPE_MASK | LC_NUMERIC_MASK, "C", nullptr);
  ASSERT_NE(loc, nullptr);
  LIBC_NAMESPACE::freelocale(loc);

  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "POSIX", nullptr);
  ASSERT_NE(loc, nullptr);
  LIBC_NAMESPACE::freelocale(loc);

  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "", nullptr);
  ASSERT_NE(loc, nullptr);
  LIBC_NAMESPACE::freelocale(loc);

  EXPECT_EQ(LIBC_NAMESPACE::newlocale(~0, "C", nullptr), nullptr);
  ASSERT_ERRNO_EQ(EINVAL);

  EXPECT_EQ(LIBC_NAMESPACE::newlocale(LC_ALL_MASK, nullptr, nullptr), nullptr);
  ASSERT_ERRNO_EQ(EINVAL);

  EXPECT_EQ(LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "does-not-exist", nullptr),
            nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

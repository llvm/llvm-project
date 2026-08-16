//===-- Unittests for locale ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/locale_macros.h"
#include "src/locale/freelocale.h"
#include "src/locale/newlocale.h"
#include "src/locale/uselocale.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcLocale, DefaultLocale) {
  locale_t new_locale = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "C", nullptr);
  ASSERT_NE(new_locale, static_cast<locale_t>(nullptr));

  locale_t old_locale = LIBC_NAMESPACE::uselocale(new_locale);
  ASSERT_NE(old_locale, static_cast<locale_t>(nullptr));

  LIBC_NAMESPACE::freelocale(new_locale);

  locale_t restored_locale = LIBC_NAMESPACE::uselocale(old_locale);
  ASSERT_NE(restored_locale, static_cast<locale_t>(nullptr));
}

TEST(LlvmLibcLocale, NewLocaleValidation) {
  locale_t loc =
      LIBC_NAMESPACE::newlocale(LC_CTYPE_MASK | LC_NUMERIC_MASK, "C", nullptr);
  ASSERT_NE(loc, static_cast<locale_t>(nullptr));
  LIBC_NAMESPACE::freelocale(loc);

  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "", nullptr);
  ASSERT_NE(loc, static_cast<locale_t>(nullptr));
  LIBC_NAMESPACE::freelocale(loc);

  loc = LIBC_NAMESPACE::newlocale(~0, "C", nullptr);
  EXPECT_EQ(loc, static_cast<locale_t>(nullptr));

  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "does-not-exist", nullptr);
  EXPECT_EQ(loc, static_cast<locale_t>(nullptr));
}

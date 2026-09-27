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
  EXPECT_NE(new_locale, static_cast<locale_t>(nullptr));

  locale_t old_locale = LIBC_NAMESPACE::uselocale(new_locale);
  EXPECT_NE(old_locale, static_cast<locale_t>(nullptr));

  LIBC_NAMESPACE::freelocale(new_locale);

  LIBC_NAMESPACE::uselocale(old_locale);
}

TEST(LlvmLibcLocale, NewLocaleValidation) {
  // Choosing masks within LC_*_MASK is OK.
  locale_t loc =
      LIBC_NAMESPACE::newlocale(LC_CTYPE_MASK | LC_NUMERIC_MASK, "C", nullptr);
  EXPECT_NE(loc, static_cast<locale_t>(nullptr));
  LIBC_NAMESPACE::freelocale(loc);

  // Empty locale name is implementation-defined,
  // defaults to C locale.
  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "", nullptr);
  EXPECT_NE(loc, static_cast<locale_t>(nullptr));
  LIBC_NAMESPACE::freelocale(loc);

  // Masks outside the valid range are rejected.
  loc = LIBC_NAMESPACE::newlocale(~0, "C", nullptr);
  EXPECT_EQ(loc, static_cast<locale_t>(nullptr));

  // Invalid locale name is rejected.
  loc = LIBC_NAMESPACE::newlocale(LC_ALL_MASK, "does-not-exist", nullptr);
  EXPECT_EQ(loc, static_cast<locale_t>(nullptr));
}

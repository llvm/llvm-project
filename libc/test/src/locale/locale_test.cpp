//===-- Unittests for locale ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/freelocale.h"
#include "src/locale/newlocale.h"
#include "src/locale/uselocale.h"

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/locale-macros.h"

TEST(LlvmLibcLocale, DefaultLocale) {
  locale_t new_locale = LIBC_NAMESPACE::newlocale(LC_ALL, "C", nullptr);
  EXPECT_NE(new_locale, static_cast<locale_t>(nullptr));

  locale_t old_locale = LIBC_NAMESPACE::uselocale(new_locale);
  EXPECT_NE(old_locale, static_cast<locale_t>(nullptr));

  LIBC_NAMESPACE::freelocale(new_locale);

  LIBC_NAMESPACE::uselocale(old_locale);
}

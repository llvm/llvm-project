//===-- Unittests for strncasecmp_l ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/locale_macros.h"
#include "src/locale/freelocale.h"
#include "src/locale/newlocale.h"
#include "src/strings/strncasecmp_l.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrNCaseCmpLTest, Case) {
  locale_t locale = LIBC_NAMESPACE::newlocale(LC_ALL, "C", nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::strncasecmp_l("hello", "HELLO", 3, locale), 0);
  ASSERT_EQ(LIBC_NAMESPACE::strncasecmp_l("abcXX", "ABCYY", 3, locale), 0);
  ASSERT_LT(LIBC_NAMESPACE::strncasecmp_l("hello1", "hello2", 6, locale), 0);
  ASSERT_GT(LIBC_NAMESPACE::strncasecmp_l("hello2", "hello1", 6, locale), 0);
  LIBC_NAMESPACE::freelocale(locale);
}

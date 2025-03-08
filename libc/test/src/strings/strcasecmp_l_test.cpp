//===-- Unittests for strcasecmp_l ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/freelocale.h"
#include "src/locale/newlocale.h"
#include "src/strings/strcasecmp_l.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCaseCmpLTest, Case) {
  locale_t locale = newlocale(LC_ALL, "C", nullptr);
  ASSERT_EQ(strcasecmp_l("hello", "HELLO", locale), 0);
  ASSERT_LT(strcasecmp_l("hello1", "hello2", locale), 0);
  ASSERT_GT(strcasecmp_l("hello2", "hello1", locale), 0);
  freelocale(locale);
}

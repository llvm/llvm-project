//===-- Unittests for localeconv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/locale-macros.h"
#include "src/locale/localeconv.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcLocale, DefaultLocale) {
  struct lconv *conv = LIBC_NAMESPACE::localeconv();
  EXPECT_STREQ(conv->decimal_point, ".");
}

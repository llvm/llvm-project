//===-- Unittests for puts ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "src/stdio/fputs.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcPutsTest, PrintOut) {
  int result;

  constexpr char simple[] = "A simple string written to stdout\n";
  result = LIBC_NAMESPACE::fputs(
      simple, reinterpret_cast<FILE *>(LIBC_NAMESPACE::stdout));
  EXPECT_GE(result, 0);

  constexpr char more[] = "A simple string written to stderr\n";
  result = LIBC_NAMESPACE::fputs(
      more, reinterpret_cast<FILE *>(LIBC_NAMESPACE::stderr));
  EXPECT_GE(result, 0);
}

//===-- Unittests for fputc / putchar -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "src/stdio/fputc.h"
#include "src/stdio/putchar.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcPutcTest, PrintOut) {
  int result;

  constexpr char simple[] = "A simple string written to stdout\n";
  for (const char &c : simple) {
    result = __llvm_libc::putchar(c);
    EXPECT_GE(result, 0);
  }

  constexpr char more[] = "A simple string written to stderr\n";
  for (const char &c : simple) {
    result =
        __llvm_libc::fputc(c, reinterpret_cast<FILE *>(__llvm_libc::stderr));
  }
  EXPECT_GE(result, 0);
}

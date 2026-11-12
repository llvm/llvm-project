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
#include "src/stdio/stderr.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcPutcTest, PrintOut) {
  int result;

  constexpr char simple[] = "A simple string written to stdout\n";
  for (const char &c : simple) {
    if (!c)
      break;
    result = LIBC_NAMESPACE::putchar(c);
    EXPECT_EQ(result, static_cast<int>(static_cast<unsigned char>(c)));
  }

  constexpr char more[] = "A simple string written to stderr\n";
  for (const char &c : more) {
    if (!c)
      break;
    result = LIBC_NAMESPACE::fputc(
        c, reinterpret_cast<FILE *>(LIBC_NAMESPACE::stderr));
    EXPECT_GE(result, 0);
  }
}

TEST(LlvmLibcPutcTest, ReturnsTheByteUnsigned) {
  // A byte above 0x7f comes back as an unsigned char, never sign extended.
  EXPECT_EQ(LIBC_NAMESPACE::fputc(
                0xff, reinterpret_cast<FILE *>(LIBC_NAMESPACE::stderr)),
            0xff);
}
